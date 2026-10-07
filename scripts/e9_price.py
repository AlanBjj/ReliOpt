"""e9, priced escalation: per question choose accept / IRCoT / Search-o1 by estimated benefit minus price x cost.

Why: the first e9 pass (scripts/e9_escalate.py) fixed the escalation share by hand; the method
needs a rule that sets operating points without test labels and does not escalate where the alternative is no better
(2WikiMultiHopQA). As for verification (e3), one price lambda trades estimated accuracy against tokens:
  benefit_A(s) = P(A right | s) - P(plan right | s)   s = the composed reliability of the plan's answer (scripts/e9_escalate.py)
  choice       = argmax over {accept: 0, A: benefit_A(s) - lambda * c_A}, A in {IRCoT (official), Search-o1}
P(. | s) are isotonic regressions on 500 calibration questions of the same backbone and dataset; c_A is the alternative's
mean tokens per question there (Search-o1: the mean over the seeds run, as on test). --calsrc devcal (default) uses
development questions disjoint from test and tune (scripts/e9_devcal.py; e2 --split devcal), which share the test subsets'
distribution; devcal2 500 more of them, dev1000 both; --calsrc cal uses e3's training
calibration questions (e2 --split cal), kept as the first attempt: it over-escalated because the training questions are
easier and favour IRCoT more than the test subsets do (2026-10-07 23:25). lambda is set for a target average total budget B (tokens per question):
  cal       bisected on the calibration questions (their plan-only tokens plus the method's measured verification spend)
  workload  bisected on the test questions' own reliabilities and the calibration costs (no test label, no test cost of an
            alternative is used: the budget holds for the workload, as e3's price holds for an average budget)
Budgets (as B_SC in e3, anchored to a baseline's measured tokens per question on that test set): B_IRCoT/2, half of
official IRCoT's, and B_SO1, Search-o1's (mean over seeds); plus a sweep for the quality-cost curve.
Accepted questions keep the e3 method at its priced lambda (B_SC); escalated ones are answered by the alternative, with
tokens = the plan's default execution + the alternative's tokens (escalation is decided before any verification).
joint: one price for verification and escalation; a question is escalated when the alternative's net
value beats the ex-ante net value of verifying it at the same price (class Verify), and the questions kept are verified
at that price (e3's rows at the nearest of its materialised prices); lambda bisected on the workload's estimated spend.
jointf: joint with the verification price floored at e3's B_SC price (more verification than that buys nothing in e3).
Ablations replace s by the other signals of scripts/e9_escalate.py; controls at every budget: random escalation with
the method's number of escalations per alternative (mean of 20 draws) and an after-the-fact oracle under the same spend
(in the joint mode both keep the point's verification price for the questions not escalated).
Input:  results/e3/<tag>/{cal,test}/<ds>/questions.jsonl, models/, prices.json, test runs; results/e2/<tag>/{cal,test}/<ds>/
Output: results/e9/<tag>/priced.{json,md} and priced_curve.csv (devcal); priced_{devcal2,dev1000,traincal}.* for the
        other --calsrc; --out-dir elsewhere
Usage:  python scripts/e9_price.py --tag llama31-8b      (project venv)
"""
import argparse
import csv
import json
import math
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from e9_escalate import DATASETS, SEEDS, rd, signals, tok  # noqa: E402

ALTS = ["ircot", "searcho1"]
# calibration sources: devcal = 500 development questions (seed 2026), devcal2 = 500 more (seed 2027, added to
# enlarge the calibration set), dev1000 = both; cal = e3's training calibration questions (the first attempt)
CALSRC = {"devcal": ["devcal"], "devcal2": ["devcal2"], "dev1000": ["devcal", "devcal2"], "cal": ["cal"]}
STEM = {"devcal": "priced", "devcal2": "priced_devcal2", "dev1000": "priced_dev1000", "cal": "priced_traincal"}
SIGS = ["plan", "noprop", "sink", "prior"]


def boot(d, n=10000, seed=0):
    """Mean and 95% paired-bootstrap interval (x100) of per-question differences d."""
    import numpy as np
    d = np.asarray(d, dtype=float)
    bs = np.sort(100 * d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1))
    return float(100 * d.mean()), float(bs[int(0.025 * n)]), float(bs[int(0.975 * n)])


class Model:
    """Isotonic P(right | s) for the plan and each alternative, and the alternatives' mean costs, on calibration questions."""

    def __init__(self, s, y_plan, y_alt, c_alt):
        from sklearn.isotonic import IsotonicRegression
        self.p_plan = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s, y_plan)
        self.p_alt = {a: IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s, y_alt[a]) for a in y_alt}
        self.c = c_alt

    def benefits(self, s):
        pp = self.p_plan.predict(s)
        return {a: [float(x - y) for x, y in zip(m.predict(s), pp)] for a, m in self.p_alt.items()}

    def choose(self, ben, lam):
        """Per question: None (accept) or the alternative with the largest benefit - lam * cost, if positive."""
        out = []
        for i in range(len(next(iter(ben.values())))):
            best, val = None, 0.0
            for a in ben:
                v = ben[a][i] - lam * self.c[a]
                if v > val:
                    best, val = a, v
            out.append(best)
        return out


class Verify:
    """Ex-ante value of step-level verification for one question at price lam, from its default execution only (the
    escalation decision comes before any operator runs): sum over steps of max(0, max_o pi S [eps rho+ - (1-eps) rho-]
    - lam c(kind, o)) as in reliopt/exec/adaptive.py, with S = prod over the other steps of (1 - eps_u pi_u) from their
    default-execution estimates (adaptive.py uses posteriors of the steps already verified) and no PROBE."""

    def __init__(self, qm, q):
        from reliopt.exec.operators import OPS
        steps = q["steps"]
        eps = qm.eps([st["feat"] for st in steps])
        pis = qm.pis([{"id": st["id"], "deps": st["deps"], "kind": st["kind"]} for st in steps])
        self.opts = []   # per step: [(gain, cost)] over its operators
        for i, st in enumerate(steps):
            w = pis.get(st["id"], 1.0) * math.prod(1 - eps[j] * pis.get(u["id"], 1.0) for j, u in enumerate(steps) if j != i)
            ops = []
            for op in OPS[st["kind"]]:
                if op in ("NONE", "PROBE"):
                    continue
                rp, rm = qm.rho.get((st["kind"], op), (0.0, 0.0))
                ops.append((w * (eps[i] * rp - (1 - eps[i]) * rm), qm.cost.get((st["kind"], op), 0.0)))
            self.opts.append(ops)

    def __call__(self, lam):
        """(net value, expected verification tokens) at price lam."""
        val = cost = 0.0
        for ops in self.opts:
            best = max(((g - lam * c, c) for g, c in ops), default=(0.0, 0.0))
            if best[0] > 0:
                val += best[0]; cost += best[1]
        return val, cost


def spend(choice, c):
    return sum(c[a] for a in choice if a) / len(choice)


def bisect(model, ben, target):
    """Smallest lambda whose estimated escalation spend per question is <= target (0 escalates whenever it helps)."""
    if spend(model.choose(ben, 0.0), model.c) <= target:
        return 0.0
    lo, hi = 0.0, 1.0
    while spend(model.choose(ben, hi), model.c) > target:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if spend(model.choose(ben, mid), model.c) > target else (lo, mid)
    return hi


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--calsrc", choices=list(CALSRC), default="devcal")
    ap.add_argument("--out-dir", default=None, help="default results/e9/<tag> (another directory for dry runs)")
    args = ap.parse_args()
    e3, e2 = ROOT / "results" / "e3" / args.tag, ROOT / "results" / "e2" / args.tag
    prices = json.loads((e3 / "prices.json").read_text())
    out, curve = {}, []
    for ds in DATASETS:
        qm = pickle.load(open(e3 / "models" / f"{ds}_logreg.pkl", "rb"))
        calq, cal_alt, cal_info = {}, {"ircot": {}, "searcho1": {}}, {}
        for split in CALSRC[args.calsrc]:
            calp = e2 / split / ds
            qfile = (e3 / "cal" if split == "cal" else ROOT / "results" / "e9" / args.tag / split) / ds / "questions.jsonl"
            so1 = sorted(calp.glob("searcho1_s*.jsonl"))
            if not (list(calp.glob("ircot_k*.jsonl")) and so1 and qfile.exists()):
                print(f"{ds}: {split} outputs missing, skipped")
                calq = {}
                break
            calq.update({q["qid"]: q for q in rd(qfile)})
            cal_alt["ircot"].update({r["qid"]: r for r in rd(next(calp.glob("ircot_k*.jsonl")))})
            # Search-o1 samples: the mean over the seeds run (test scores Search-o1 as the mean over seeds 17 / 29 / 43)
            runs = [{r["qid"]: r for r in rd(f)} for f in so1]
            for q in set.intersection(*(set(r) for r in runs)):
                cal_alt["searcho1"][q] = {"em": sum(r[q]["em"] for r in runs) / len(runs),
                                          "tok": sum(tok(r[q]) for r in runs) / len(runs)}
            cal_info[split] = {"questions": len(calq), "searcho1_seeds": [f.stem.split("_s")[-1] for f in so1]}
        if not calq:
            continue
        cq = sorted(set(calq) & set(cal_alt["ircot"]) & set(cal_alt["searcho1"]))
        csig = {q: signals(qm, calq[q]) for q in cq}
        c_alt = {a: sum(tok(cal_alt[a][q]) for q in cq) / len(cq) for a in ALTS}
        # test side
        lam_v = prices[ds]["B_SC"]
        po = {r["qid"]: r for r in rd(e3 / "test" / ds / "runs" / "plan_only_s17.jsonl")}
        full = {s: {} for s in SEEDS}   # seed -> e3 price lambda_v -> qid -> the method's row at that price
        for s in SEEDS:
            for r in rd(e3 / "test" / ds / "runs" / f"full_s{s}.jsonl"):
                full[s].setdefault(r["lam"], {})[r["qid"]] = r
        grid = sorted(full[SEEDS[0]])
        ours = {s: full[s][min(grid, key=lambda g: abs(g - lam_v))] for s in SEEDS}

        def vlevel(lam):
            """The materialised verification price nearest to lam (log scale; 0 maps to 0)."""
            pos = [g for g in grid if g > 0]
            if lam <= 0 and 0.0 in grid:
                return 0.0
            return min(pos, key=lambda g: abs(math.log(g) - math.log(max(lam, 1e-12))))
        irc = {r["qid"]: r for r in rd(next((e2 / "test" / ds).glob("ircot_k*.jsonl")))}
        alt = {"ircot": {s: irc for s in SEEDS},
               "searcho1": {s: {r["qid"]: r for r in rd(e2 / "test" / ds / f"searcho1_s{s}.jsonl")} for s in SEEDS}}
        tsig = {q["qid"]: signals(qm, q) for q in rd(e3 / "test" / ds / "questions.jsonl")}
        tq = sorted(set(tsig) & set(po) & set(irc))
        own = sum(tok(ours[s][q]) for s in SEEDS for q in tq) / (len(tq) * len(SEEDS))   # the method's own cost
        b_half = 0.5 * sum(tok(irc[q]) for q in tq) / len(tq)
        b_so1 = sum(tok(alt["searcho1"][s][q]) for s in SEEDS for q in tq) / (len(tq) * len(SEEDS))

        def run(choice, vr=None):
            vr = vr or ours   # rows of the questions that are not escalated (the method at a verification price)
            em = f1 = tk = 0.0
            per_q = {}
            for s in SEEDS:
                for q, a in zip(tq, choice):
                    r = alt[a][s][q] if a else vr[s][q]
                    em += r["em"]; f1 += r["f1"]
                    tk += (tok(po[q]) + tok(r)) if a else tok(r)
                    per_q[q] = per_q.get(q, 0) + r["em"] / len(SEEDS)
            n = len(tq) * len(SEEDS)
            return {"em": 100 * em / n, "f1": 100 * f1 / n, "tok": tk / n, "per_q": per_q,
                    "share": {a: sum(1 for c in choice if c == a) / len(tq) for a in ALTS},
                    "choice": {q: a for q, a in zip(tq, choice) if a}}

        tqs = {q["qid"]: q for q in rd(e3 / "test" / ds / "questions.jsonl")}
        vfun = {q: Verify(qm, tqs[q]) for q in tq}
        res = {"cal": cal_info, "n_cal": len(cq), "own_tok": own, "B_IRCoT/2": b_half, "B_SO1": b_so1, "c_alt_cal": c_alt, "points": {}}
        for sig in SIGS:
            model = Model([csig[q][sig] for q in cq], [calq[q]["base"]["em"] for q in cq],
                          {a: [cal_alt[a][q]["em"] for q in cq] for a in ALTS}, c_alt)
            cben = model.benefits([csig[q][sig] for q in cq])
            tben = model.benefits([tsig[q][sig] for q in tq])
            for mode, ben in (("cal", cben), ("workload", tben)):
                for name, B in [("B_IRCoT/2", b_half), ("B_SO1", b_so1)] + [(f"E{e}", own + e) for e in (0, 1000, 2000, 4000, 8000, 12000, 16000, 24000)]:
                    lam = bisect(model, ben, max(B - own, 0.0))
                    r = run(model.choose(tben, lam))
                    r["lam"] = lam
                    key = f"{sig}/{mode}/{name}"
                    res["points"][key] = r
                    curve.append([args.tag, ds, sig, mode, name, lam, round(r["em"], 2), round(r["f1"], 2), round(r["tok"], 1),
                                  round(r["share"]["ircot"], 3), round(r["share"]["searcho1"], 3)])
            # joint pricing: one price lambda
            # for both decisions. Per question: escalate to A when benefit_A(s) - lambda c_A exceeds the ex-ante net value
            # of verifying at lambda (class Verify), else keep the plan and verify at lambda (e3's rows at the nearest
            # materialised price). lambda is bisected on the test workload's estimated total spend (default execution, as
            # measured, + c_A or the expected verification tokens; no test label), budgets as above.
            def joint(lam, floor=0.0):
                ch, sp = [], 0.0
                for i, q in enumerate(tq):
                    best, (val, cost) = None, vfun[q](max(lam, floor))
                    for a in ALTS:
                        v = tben[a][i] - lam * c_alt[a]
                        if v > val:
                            best, val, cost = a, v, c_alt[a]
                    ch.append(best)
                    sp += tok(po[q]) + cost
                return ch, sp / len(tq)
            # jointf: the same, with verification never priced below e3's own calibrated price (B_SC): beyond it the quality
            # model's repair estimates leave its calibrated range and e3's curve flattens or dips (more verification buys
            # nothing), so a low price may only buy more escalation
            for jmode, floor in (("joint", 0.0), ("jointf", lam_v)):
                for name, B in [("B_IRCoT/2", b_half), ("B_SO1", b_so1)] + [(f"E{e}", own + e) for e in (0, 1000, 2000, 4000, 8000, 12000, 16000, 24000)]:
                    if joint(0.0, floor)[1] <= B:
                        lam = 0.0
                    else:
                        lo, hi = 0.0, 1e-6
                        while joint(hi, floor)[1] > B:
                            hi *= 2
                        for _ in range(40):
                            mid = (lo + hi) / 2
                            lo, hi = (mid, hi) if joint(mid, floor)[1] > B else (lo, mid)
                        lam = hi
                    lv = vlevel(max(lam, floor))
                    r = run(joint(lam, floor)[0], {s: full[s][lv] for s in SEEDS})
                    r["lam"], r["vlam"] = lam, lv
                    res["points"][f"{sig}/{jmode}/{name}"] = r
                    curve.append([args.tag, ds, sig, jmode, name, lam, round(r["em"], 2), round(r["f1"], 2), round(r["tok"], 1),
                                  round(r["share"]["ircot"], 3), round(r["share"]["searcho1"], 3)])
        # controls at every budget (the working point is chosen after the fact, so each one needs its own): the same number
        # of escalations per alternative as the method, given to random questions (mean of 20 draws), and an after-the-fact
        # oracle that escalates by realised gain under the same spend
        budgets = sorted({k.split("/", 2)[2] for k in res["points"] if k.startswith("plan/")})
        for mode, bname in [(m, b) for m in ("cal", "workload", "joint", "jointf") for b in budgets]:
            main = res["points"][f"plan/{mode}/{bname}"]
            vr = {s: full[s][main["vlam"]] for s in SEEDS} if mode.startswith("joint") else ours
            base = sum(tok(vr[s][q]) for s in SEEDS for q in tq) / (len(tq) * len(SEEDS))
            counts = {a: round(main["share"][a] * len(tq)) for a in ALTS}
            draws = []
            for i in range(20):
                order = tq[:]; random.Random(i).shuffle(order)
                ch = {}
                k = 0
                for a in ALTS:
                    for q in order[k:k + counts[a]]:
                        ch[q] = a
                    k += counts[a]
                draws.append(run([ch.get(q) for q in tq], vr))
            res["points"][f"random/{mode}/{bname}"] = {m: sum(d[m] for d in draws) / len(draws) for m in ("em", "f1", "tok")}
            res["points"][f"random/{mode}/{bname}"]["share"] = main["share"]
            gains = []
            for q in tq:
                for a in ALTS:
                    g = sum(alt[a][s][q]["em"] - vr[s][q]["em"] for s in SEEDS) / len(SEEDS)
                    c = tok(po[q]) + sum(tok(alt[a][s][q]) for s in SEEDS) / len(SEEDS) - sum(tok(vr[s][q]) for s in SEEDS) / len(SEEDS)
                    if g > 0:
                        gains.append((g / max(c, 1.0), q, a, c))
            budget, used, ch = max(main["tok"] - base, 0.0) * len(tq), 0.0, {}
            for _, q, a, c in sorted(gains, reverse=True):
                if q not in ch and used + c <= budget:
                    ch[q] = a; used += c
            r = run([ch.get(q) for q in tq], vr); r.pop("per_q", None)
            res["points"][f"oracle/{mode}/{bname}"] = r
        # references for the bootstrap
        refs = {"ours": {q: sum(ours[s][q]["em"] for s in SEEDS) / len(SEEDS) for q in tq},
                "ircot": {q: irc[q]["em"] for q in tq},
                "searcho1": {q: sum(alt["searcho1"][s][q]["em"] for s in SEEDS) / len(SEEDS) for q in tq}}
        for key in [k for k in res["points"] if k.split("/")[0] in SIGS]:   # every signal, mode and budget
            pq = res["points"][key]["per_q"]
            res["points"][key]["boot"] = {k: boot([pq[q] - v[q] for q in tq]) for k, v in refs.items()}
            if not key.startswith("plan/"):   # ablated signal against the method's reliability at the same mode and budget
                pp = res["points"]["plan/" + key.split("/", 1)[1]]["per_q"]
                res["points"][key]["boot"]["esc"] = boot([pq[q] - pp[q] for q in tq])
        res["refs"] = {k: 100 * sum(v.values()) / len(v) for k, v in refs.items()}
        for k, p in res["points"].items():
            p.pop("per_q", None)
            if not (k.startswith("plan/") and k.split("/", 2)[2] in ("B_IRCoT/2", "B_SO1")):
                p.pop("choice", None)   # escalated questions kept only at the two main budgets (scripts/e9_escalated.py)
        out[ds] = res
    base = Path(args.out_dir) if args.out_dir else ROOT / "results" / "e9" / args.tag
    base.mkdir(parents=True, exist_ok=True)
    stem = STEM[args.calsrc]
    (base / f"{stem}.json").write_text(json.dumps(out, indent=1))
    with open(base / f"{stem}_curve.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "dataset", "signal", "mode", "budget", "lambda", "em", "f1", "tok", "share_ircot", "share_searcho1"])
        w.writerows(curve)
    md = [f"# e9 priced escalation ({args.tag}, test; escalation maps fitted on {args.calsrc})", ""]
    for ds, r in out.items():
        md += [f"## {ds}", "", f"refs EM: " + ", ".join(f"{k} {v:.1f}" for k, v in r["refs"].items())
               + f"; method's own tokens {r['own_tok']:,.0f}; B_IRCoT/2 = {r['B_IRCoT/2']:,.0f}; {r['n_cal']} calibration "
               + "questions (" + "; ".join(f"{k}: Search-o1 seeds {', '.join(v['searcho1_seeds'])}" for k, v in r["cal"].items())
               + "); calibration costs "
               + ", ".join(f"{a} {c:,.0f}" for a, c in r["c_alt_cal"].items()), "",
               "| signal / mode / budget | EM | F1 | tokens | IRCoT share | Search-o1 share | vs ours | vs IRCoT | vs Search-o1 | vs plan signal |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for key, p in r["points"].items():
            b = p.get("boot", {})
            fmt = lambda k: f"{b[k][0]:+.1f} [{b[k][1]:+.1f}, {b[k][2]:+.1f}]" if k in b else ""
            sh = p.get("share", {"ircot": 0, "searcho1": 0})
            md.append(f"| {key} | {p['em']:.1f} | {p['f1']:.1f} | {p['tok']:,.0f} | {100 * sh['ircot']:.0f}% | "
                      f"{100 * sh['searcho1']:.0f}% | {fmt('ours')} | {fmt('ircot')} | {fmt('searcho1')} | {fmt('esc')} |")
        md.append("")
    (base / f"{stem}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
