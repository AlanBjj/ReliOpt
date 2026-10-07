"""e9: reliability-triggered escalation from the typed plan to an expensive physical plan (IRCoT / Search-o1).

Why: step-level verification adds little because the operators rarely repair an error, but the
quality model predicts errors well and the typed plan and iterative retrieval fail on different questions (answered by at
least one of them: about 59 / 67 / 32-36 EM against 41-46 / 52-56 / 21-27 for either alone). The method therefore also
gets a plan-level physical alternative, as in progressive query optimisation: after the plan's default execution the
composed reliability of the answer is checked, and a low-reliability question is re-executed with an expensive strategy
instead of being verified step by step.

Decision (made right after the default execution, before any verification; no gold answer is used at test time):
  accept    the question continues with the step-level allocator of e3 at its priced lambda (B_SC): its answer and tokens
  escalate  the question is answered by the alternative; tokens = the plan's default execution (already spent) + the
            alternative's tokens
Signals (lower = less reliable = escalated first), all computed from the plan-only execution features:
  plan      prod_v (1 - eps_v * pi_v)      the method: instance estimates composed along the plan (propagation weights)
  noprop    prod_v (1 - eps_v)              ablation: no propagation weights
  sink      1 - eps_sink                    ablation: only the step that gives the answer (an answer-confidence router)
  prior     prod_v (1 - epsbar_kind * pi_v) ablation: type priors instead of instance estimates
  random    mean of 20 random orders        control
  oracle    escalate where the alternative is right and the method wrong (upper bound, uses gold answers)
Operating points without test labels: an escalation fraction f fixes the threshold as the f-quantile of the signal on the
500 calibration questions of the same backbone and dataset; the threshold is then applied to the test questions (the
realised fraction is reported).
Input:  results/e3/<tag>/{cal,test}/<ds>/questions.jsonl, results/e3/<tag>/models/<ds>_logreg.pkl, prices.json,
        results/e3/<tag>/test/<ds>/runs/{plan_only_s17,full_s<seed>}.jsonl,
        results/e2/<tag>/test/<ds>/{ircot_k<K>,searcho1_s<seed>}.jsonl
Output: results/e9/<tag>/summary.{json,md}, results/e9/<tag>/curves.csv
Usage:  python scripts/e9_escalate.py --tag llama31-8b      (project venv: unpickles the scikit-learn quality model)
"""
import argparse
import ast
import csv
import json
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DATASETS = ["hotpotqa", "2wikimultihopqa", "musique"]
SEEDS = [17, 29, 43]
FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
SIGNALS = ["plan", "noprop", "sink", "prior", "random", "oracle"]


def rd(p):
    return [json.loads(l) for l in open(p)]


def tok(r):
    return sum(r["tok"].values()) if isinstance(r["tok"], dict) else r["tok"]


def signals(qm, q):
    """Reliability signals of one question from its plan-only execution (the rows of questions.jsonl)."""
    steps = q["steps"]
    eps = qm.eps([s["feat"] for s in steps])
    pis = qm.pis([{"id": s["id"], "deps": s["deps"], "kind": s["kind"]} for s in steps])
    plan = noprop = prior = 1.0
    for s, e in zip(steps, eps):
        p = pis.get(s["id"], 1.0)
        plan *= 1 - e * p
        noprop *= 1 - e
        prior *= 1 - qm.prior(s["kind"]) * p
    return {"plan": plan, "noprop": noprop, "sink": 1 - eps[-1], "prior": prior}


def quantile(xs, f):
    xs = sorted(xs)
    return xs[min(int(f * len(xs)), len(xs) - 1)] if f > 0 else float("-inf")


def evaluate(qs, esc, ours, alt):
    """Mean over seeds of EM / F1 / tokens when the questions in `esc` are escalated. ours/alt: {seed: {qid: row}}."""
    em = f1 = tk = 0.0
    for s in SEEDS:
        for q in qs:
            if q in esc:
                r = alt[s][q]
                em += r["em"]; f1 += r["f1"]; tk += ours["plan_tok"][q] + tok(r)
            else:
                r = ours[s][q]
                em += r["em"]; f1 += r["f1"]; tk += tok(r)
    n = len(qs) * len(SEEDS)
    return {"em": 100 * em / n, "f1": 100 * f1 / n, "tok": tk / n, "frac": len(esc) / len(qs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    e3, e2 = ROOT / "results" / "e3" / args.tag, ROOT / "results" / "e2" / args.tag / "test"
    prices = json.loads((e3 / "prices.json").read_text())
    out, rows_csv = {}, []
    for ds in DATASETS:
        qm = pickle.load(open(e3 / "models" / f"{ds}_logreg.pkl", "rb"))
        cal = [signals(qm, q) for q in rd(e3 / "cal" / ds / "questions.jsonl")]
        test = {q["qid"]: signals(qm, q) for q in rd(e3 / "test" / ds / "questions.jsonl")}
        lam = prices[ds]["B_SC"]
        po = {r["qid"]: r for r in rd(e3 / "test" / ds / "runs" / "plan_only_s17.jsonl")}
        ours = {"plan_tok": {q: tok(r) for q, r in po.items()}}
        for s in SEEDS:
            ours[s] = {r["qid"]: r for r in rd(e3 / "test" / ds / "runs" / f"full_s{s}.jsonl") if abs(r["lam"] - lam) < 1e-12}
        irc = {r["qid"]: r for r in rd(next(e2.glob(f"{ds}/ircot_k*.jsonl")))}
        alts = {"ircot": {s: irc for s in SEEDS},
                "searcho1": {s: {r["qid"]: r for r in rd(e2 / ds / f"searcho1_s{s}.jsonl")} for s in SEEDS}}
        qs = sorted(set(test) & set(po) & set(irc) & set.intersection(*(set(ours[s]) for s in SEEDS)))
        y = [po[q]["em"] for q in qs]
        res = {"n": len(qs), "auroc": {k: roc_auc_score(y, [test[q][k] for q in qs]) for k in ("plan", "noprop", "sink", "prior")},
               "ours": evaluate(qs, set(), ours, alts["ircot"]), "curves": {}}
        for a, alt in alts.items():
            res["alt_" + a] = evaluate(qs, set(qs), ours, alt)
            gain = {q: sum(alt[s][q]["em"] - ours[s][q]["em"] for s in SEEDS) for q in qs}
            for sig in SIGNALS:
                pts = []
                for f in FRACS:
                    if sig in ("plan", "noprop", "sink", "prior"):
                        t = quantile([c[sig] for c in cal], f)
                        esc = {q for q in qs if test[q][sig] < t}
                        pts.append(evaluate(qs, esc, ours, alt))
                    elif sig == "random":
                        k = int(round(f * len(qs)))
                        runs = []
                        for i in range(20):
                            order = qs[:]; random.Random(i).shuffle(order)
                            runs.append(evaluate(qs, set(order[:k]), ours, alt))
                        pts.append({m: sum(r[m] for r in runs) / len(runs) for m in runs[0]})
                    else:   # oracle: largest gains first, same number of questions as f
                        k = int(round(f * len(qs)))
                        order = sorted(qs, key=lambda q: -gain[q])
                        pts.append(evaluate(qs, set(order[:k]), ours, alt))
                    pts[-1]["f"] = f
                    rows_csv.append([args.tag, ds, a, sig, f, round(pts[-1]["frac"], 4), round(pts[-1]["em"], 2),
                                     round(pts[-1]["f1"], 2), round(pts[-1]["tok"], 1)])
                res["curves"][f"{a}/{sig}"] = pts
        out[ds] = res
    base = ROOT / "results" / "e9" / args.tag
    base.mkdir(parents=True, exist_ok=True)
    (base / "summary.json").write_text(json.dumps(out, indent=1))
    with open(base / "curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "dataset", "alternative", "signal", "f", "realised_frac", "em", "f1", "tok"])
        w.writerows(rows_csv)
    md = [f"# e9 escalation ({args.tag}, test; thresholds from calibration quantiles)", ""]
    for ds, r in out.items():
        md += [f"## {ds}", "",
               f"method at B_SC: EM {r['ours']['em']:.1f} / {r['ours']['tok']:,.0f} tok; IRCoT {r['alt_ircot']['em']:.1f} / "
               f"{r['alt_ircot']['tok']:,.0f} (incl. plan); Search-o1 {r['alt_searcho1']['em']:.1f} / {r['alt_searcho1']['tok']:,.0f}",
               "AUROC for plan-only correctness: " + ", ".join(f"{k} {v:.3f}" for k, v in r["auroc"].items()), "",
               "| alternative / signal | " + " | ".join(f"f={f:.1f}" for f in FRACS) + " |",
               "|---|" + "---|" * len(FRACS)]
        for key, pts in r["curves"].items():
            md.append(f"| {key} | " + " | ".join(f"{p['em']:.1f} / {p['tok'] / 1000:.1f}k ({100 * p['frac']:.0f}%)" for p in pts) + " |")
        md.append("")
    (base / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
