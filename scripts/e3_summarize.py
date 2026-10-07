"""e3 step 5: summarise the method and the internal comparisons.

  curves     per variant: EM / F1 / verification and total tokens at every lambda (Type-Static: every calibration-frontier
             policy), mean over seeds
  EM@B       for B in {B_SC, B_SC/2} (verification tokens, see scripts/e3_price.py): the variant's EM interpolated
             linearly between the two operating points whose measured verification costs bracket B (a mixture of the two
             points; Plan-Only is the zero-cost point); points are placed by cost only, never chosen by EM
  AUC-QC     mean interpolated EM over verification budgets 0..Verify-All's cost, i.e. the area under the quality-cost
             curve with cost normalised by Verify-All
  main       the method at the priced lambdas: EM / F1 (mean ± std over seeds), tokens per question, Cost (x) against
             Standard RAG of results/e2
  oracle     per-question best cached single-step intervention at each price (needs the test materialisation)
  diagnostics eps-hat AUROC and ECE on the labelled test steps (diagnostics only, never fitted on)
Input:  results/e3/<tag>/test/<ds>/runs/*.jsonl, results/e3/<tag>/test/<ds>/questions.jsonl, models/, prices.json,
        results/e2/<tag>/test/summary.json
Output: results/e3/<tag>/test/summary.{json,md}
Usage:  python scripts/e3_summarize.py --tag llama31-8b [--split test]
"""
import argparse
import json
import pickle
import statistics
import sys
import warnings
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from e1_analyze import oracle_frontier  # noqa: E402

warnings.filterwarnings("ignore")
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
ORDER = ["plan_only", "verify_all", "type_static", "question_level", "full", "no_prop", "prior_eps", "no_probe", "gbdt"]


def read(p):
    return [json.loads(l) for l in open(p)]


def points(rows_by_seed, key):
    """Average over seeds of each operating point (grouped by `key`): em, f1, verify and total tokens."""
    per = defaultdict(list)
    for rows in rows_by_seed:
        grp = defaultdict(list)
        ok = [r for r in rows if "error" not in r]
        for r in ok:
            grp[r.get(key)].append(r)
        for r in rows:   # a question that failed counts as wrong, at zero cost, at every operating point
            if "error" in r:
                for k in grp:
                    grp[k].append({"qid": r["qid"], "em": 0.0, "f1": 0.0})
        for k, rs in grp.items():
            n = len(rs)   # failed questions count as wrong, at zero cost
            per[k].append({"em": 100 * sum(r.get("em", 0) for r in rs) / n, "f1": 100 * sum(r.get("f1", 0) for r in rs) / n,
                           "verify_tok": sum(r["tok"]["verify"] for r in rs if "tok" in r) / n,
                           "total_tok": sum(sum(r["tok"].values()) for r in rs if "tok" in r) / n})
    out = []
    for k, ps in per.items():
        p = {m: statistics.mean(x[m] for x in ps) for m in ps[0]}
        p["em_std"] = statistics.stdev([x["em"] for x in ps]) if len(ps) > 1 else 0.0
        p["f1_std"] = statistics.stdev([x["f1"] for x in ps]) if len(ps) > 1 else 0.0
        out.append({key: k, **p})
    return sorted(out, key=lambda p: p["verify_tok"])


def em_at(pts, budget, base):
    """EM at a verification budget: linear interpolation between the two operating points that bracket it (a mixture of
    the two), with Plan-Only as the zero-cost point and the most expensive point held beyond its cost."""
    curve = sorted([(0.0, base)] + [(p["verify_tok"], p["em"]) for p in pts])
    if budget >= curve[-1][0]:
        return curve[-1][1]
    for (c0, e0), (c1, e1) in zip(curve, curve[1:]):
        if c0 <= budget <= c1:
            return e0 if c1 == c0 else e0 + (e1 - e0) * (budget - c0) / (c1 - c0)
    return base


def auc(pts, base, top, n=200):
    """Mean of the interpolated EM over verification budgets 0..top (top = Verify-All's cost)."""
    xs = [top * i / n for i in range(n + 1)]
    return statistics.mean(em_at(pts, x, base) for x in xs)


def ece(p, y, bins=10):
    tot = 0.0
    for b in range(bins):
        idx = [i for i, x in enumerate(p) if b / bins <= x < (b + 1) / bins or (b == bins - 1 and x == 1.0)]
        if idx:
            tot += len(idx) / len(p) * abs(statistics.mean(p[i] for i in idx) - statistics.mean(y[i] for i in idx))
    return tot


def diagnostics(base, ds, split):
    from sklearn.metrics import roc_auc_score
    q = base / split / ds / "questions.jsonl"
    if not q.exists():
        return {}
    rows = read(q)
    qm = pickle.load(open(base / "models" / f"{ds}_logreg.pkl", "rb"))
    st = [s for r in rows for s in r["steps"] if s["label"] in ("correct", "error")]
    y = [int(s["label"] == "error") for s in st]
    p = qm.eps([s["feat"] for s in st])
    n_all = sum(len(r["steps"]) for r in rows)
    return {"auroc": roc_auc_score(y, p), "ece": ece(p, y), "n_steps": len(st), "undet_pct": 100 * (1 - len(st) / n_all),
            "oracle": oracle_frontier(rows, False)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    base = ROOT / "results" / "e3" / args.tag
    e2 = json.loads((ROOT / "results" / "e2" / args.tag / "test" / "summary.json").read_text())["summary"]
    prices = json.loads((base / "prices.json").read_text()) if (base / "prices.json").exists() else {}
    out = {}
    for ds in DATASETS:
        rd = base / args.split / ds / "runs"
        if not rd.is_dir():
            continue
        by_var = defaultdict(list)
        for p in sorted(rd.glob("*_s*.jsonl")):
            by_var[p.stem.rsplit("_s", 1)[0]].append(read(p))
        res = {"curves": {}}
        for v, runs in by_var.items():
            res["curves"][v] = points(runs, "policy" if v in ("plan_only", "verify_all", "type_static") else "lam")
        if "plan_only" not in res["curves"]:
            print(f"{ds}: no plan_only run yet, skipped"); continue
        po = res["curves"]["plan_only"][0]
        va = res["curves"].get("verify_all", [None])[0]
        top = va["verify_tok"] if va else max(p["verify_tok"] for c in res["curves"].values() for p in c)
        b_sc = e2[ds]["sc5"]["tok"]
        res["budgets"] = {"B_SC": b_sc, "B_SC/2": b_sc / 2}
        diag = diagnostics(base, ds, args.split)
        if diag.get("oracle"):
            res["curves"]["oracle"] = diag.pop("oracle")
        res["diagnostics"] = diag
        res["table2"] = {v: {**{f"EM@{b}": em_at(c, B, po["em"]) for b, B in res["budgets"].items()},
                             "AUC-QC": auc(c, po["em"], top)} for v, c in res["curves"].items()}
        res["main"] = {}
        for name, lam in prices.get(ds, {}).items():
            p = next((p for p in res["curves"].get("full", []) if abs(p["lam"] - lam) <= 1e-12 * max(1, lam)), None)
            if p:
                res["main"][name] = {**p, "cost_x": p["total_tok"] / e2[ds]["rag"]["tok"]}
        out[ds] = res
    (base / args.split).mkdir(parents=True, exist_ok=True)
    (base / args.split / "summary.json").write_text(json.dumps(out, indent=1))
    lines = [f"# e3 ({args.tag}, {args.split})", "", "## Main table points (the method at the priced lambdas)", "",
             "| Dataset | Budget | EM | F1 | verify tok | total tok | Cost (×) |", "|---|---|---|---|---|---|---|"]
    for ds, r in out.items():
        for b, p in r["main"].items():
            lines.append(f"| {ds} | {b} = {r['budgets'][b]:.0f} | {p['em']:.1f} ± {p['em_std']:.1f} | {p['f1']:.1f} ± "
                         f"{p['f1_std']:.1f} | {p['verify_tok']:.0f} | {p['total_tok']:.0f} | {p['cost_x']:.2f} |")
    lines += ["", "## Table 2: allocation comparisons (EM@B with B = verification tokens; AUC-QC up to Verify-All's cost)",
              "", "| Variant | " + " | ".join(f"{ds} EM@B_SC | EM@B_SC/2 | AUC-QC" for ds in out) + " |",
              "|---" * (1 + 3 * len(out)) + "|"]
    for v in ORDER + ["oracle"]:
        cells = []
        for ds, r in out.items():
            t = r["table2"].get(v)
            cells += [f"{t['EM@B_SC']:.1f}", f"{t['EM@B_SC/2']:.1f}", f"{t['AUC-QC']:.1f}"] if t else ["—"] * 3
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    lines += ["", "## Operating points (verification tokens / EM, mean over seeds)", ""]
    for ds, r in out.items():
        for v in ORDER + ["oracle"]:
            c = r["curves"].get(v)
            if c:
                pts = c if v != "oracle" else c[::max(1, len(c) // 12)]
                lines.append(f"- {ds} {v}: " + ", ".join(f"{p['verify_tok']:.0f}/{p['em']:.1f}" for p in pts))
    lines += ["", "## Diagnostics", "", "| Dataset | eps AUROC | eps ECE | labelled steps | undecidable % |",
              "|---|---|---|---|---|"]
    for ds, r in out.items():
        d = r["diagnostics"]
        if d:
            lines.append(f"| {ds} | {d['auroc']:.3f} | {d['ece']:.3f} | {d['n_steps']} | {d['undet_pct']:.1f} |")
    (base / args.split / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
