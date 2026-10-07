"""Summarise the e2 baseline runs: EM / F1 / Acc / tokens per method and dataset, mean ± std over seeds, Cost (×).

Self-Consistency SC-k (k = 1..10) and Adaptive-Consistency are replayed offline from the 10-sample pools in simple_s<seed>.jsonl
(reliopt/baselines/simple.py). Cost (×) = tokens per question relative to Standard RAG on the same dataset; the "avg"
column averages the per-dataset ratios (main-table convention).
Input:  results/e2/<tag>/<split>/<ds>/*.jsonl (scripts/e2_baselines.py)
Output: results/e2/<tag>/<split>/summary.{json,md}
Usage:  python scripts/e2_summarize.py --tag llama31-8b [--split test]
Standard library only.
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reliopt.baselines.simple import adaptive_consistency, majority  # noqa: E402
from reliopt.eval.metrics import score  # noqa: E402

DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
ORDER = ["cot", "rag", "sc5", "sc10", "ac", "selfask", "ircot", "searcho1", "pyrag", "planbudget", "cove"]
TUNED = ("ircot", "planbudget")   # configuration chosen on the tuning split (IRCoT's K, Plan-and-Budget's retrieval)
NAMES = {"cot": "CoT", "rag": "Standard RAG", "sc5": "Self-Consistency (k=5)", "sc10": "Self-Consistency (k=10)",
         "ac": "Adaptive-Consistency", "selfask": "Self-Ask", "ircot": "IRCoT", "searcho1": "Search-o1", "pyrag": "PyRAG",
         "planbudget": "Plan-and-Budget", "cove": "CoVe (appendix)"}


def read(p):
    return [json.loads(l) for l in p.open()]


def stats(runs):
    """runs: list (one per seed) of per-question dicts with em/f1/acc/tok -> mean and std (over seeds) of each metric."""
    out = {"seeds": len(runs), "n": len(runs[0])}
    for k in ("em", "f1", "acc", "tok"):
        per = [sum(r[k] for r in run) / len(run) * (1 if k == "tok" else 100) for run in runs]
        out[k] = statistics.mean(per)
        out[k + "_std"] = statistics.stdev(per) if len(per) > 1 else 0.0
    for k in ("calls", "samples"):
        if k in runs[0][0]:
            out[k] = statistics.mean(sum(r[k] for r in run) / len(run) for run in runs)
    return out


def replay_pool(rows, golds, k=None):
    """SC-k (k given) or AC (k None) over each row's pool -> per-question dicts."""
    out = []
    for r in rows:
        ans, comps = r["sc_answers"], r["sc_completions"]
        prompt = r["sc10"]["tok"] - sum(comps[:10])
        if k is None:
            pred, m, tok = adaptive_consistency(ans, comps, prompt)
        else:
            pred, m, tok = majority(ans[:k]), k, prompt + sum(comps[:k])
        out.append({"pred": pred, **score(pred, golds[r["qid"]]), "tok": tok, "samples": m})
    return out


def summarize_ds(d, golds):
    res, curve = {}, {}
    simple = sorted(d.glob("simple_s*.jsonl"))
    if simple:
        pools = [read(p) for p in simple]
        res["cot"] = stats([[r["cot"] for r in pools[0]]])
        res["rag"] = stats([[r["rag"] for r in pools[0]]])
        for k in range(1, 11):
            curve[k] = stats([replay_pool(rows, golds, k) for rows in pools])
        res["sc5"], res["sc10"] = curve[5], curve[10]
        res["ac"] = stats([replay_pool(rows, golds) for rows in pools])
    for m in ("selfask", "ircot", "searcho1", "pyrag", "planbudget", "cove"):   # one entry per configuration, seeds pooled
        groups = {}
        for p in sorted(d.glob(f"{m}*.jsonl")):
            groups.setdefault(re.sub(r"_s\d+$", "", p.stem), []).append(p)
        for cfg, files in groups.items():
            res[cfg] = stats([read(p) for p in files])
    return res, curve


def gold_answers(ds, split):
    from reliopt.data.ircot import load_questions   # noqa: E402  (only needed for the pool replay)
    ids = (ROOT / "results" / "e0" / "ids" / f"{ds}_{split}.txt").read_text().split()
    return {q["qid"]: q["answers"] for q in load_questions(ds, {"test": "test_subsampled", "tune": "dev_subsampled"}[split],
                                                            qids=ids)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    base = ROOT / "results" / "e2" / args.tag / args.split
    summary, curves = {}, {}
    for ds in DATASETS:
        if (base / ds).is_dir():
            summary[ds], curves[ds] = summarize_ds(base / ds, gold_answers(ds, args.split))
    if args.split == "test":   # only the configuration selected on the tuning split was run: report it under the method
        for s in summary.values():
            for m in TUNED:
                cfgs = [c for c in s if c.startswith(m + "_")]
                if len(cfgs) == 1:
                    s[m] = {**s.pop(cfgs[0]), "cfg": cfgs[0][len(m) + 1:]}
    methods = [m for m in ORDER if any(m in s for s in summary.values())]
    methods += sorted({c for s in summary.values() for c in s if c not in ORDER})
    for s in summary.values():
        if "rag" in s:
            for m in s:
                s[m]["cost_x"] = s[m]["tok"] / s["rag"]["tok"]
    lines = [f"# e2 baselines ({args.tag}, {args.split})", "",
             "EM / F1 in %, mean ± std over seeds where sampled; Tok = tokens per question; Cost (×) = Tok / Standard RAG "
             "(avg = mean of the per-dataset ratios).", "",
             "| Method | " + " | ".join(f"{ds} EM | F1 | Tok" for ds in summary) + " | Cost (×) avg |",
             "|---" * (2 + 3 * len(summary)) + "|"]
    for m in methods:
        cells, ratios = [], []
        for ds, s in summary.items():
            if m not in s:
                cells += ["—", "—", "—"]
                continue
            r = s[m]
            pm = lambda k: f"{r[k]:.1f}" + (f" ± {r[k + '_std']:.1f}" if r["seeds"] > 1 else "")
            cells += [pm("em") + (f" ({r['cfg']})" if "cfg" in r else ""), pm("f1"), f"{r['tok']:,.0f}"]
            if "cost_x" in r:
                ratios.append(r["cost_x"])
        avg = f"{statistics.mean(ratios):.2f}" if len(ratios) == len(summary) else "—"
        lines.append(f"| {NAMES.get(m, m)} | " + " | ".join(cells) + f" | {avg} |")
    if any(curves.values()):
        lines += ["", "## Self-Consistency curve (EM / Tok by k) and Adaptive-Consistency", "",
                  "| k | " + " | ".join(summary) + " |", "|---" * (1 + len(summary)) + "|"]
        for k in range(1, 11):
            lines.append(f"| {k} | " + " | ".join(f"{curves[ds][k]['em']:.1f} / {curves[ds][k]['tok']:,.0f}"
                                                   if k in curves[ds] else "—" for ds in summary) + " |")
        lines.append("| AC | " + " | ".join(f"{summary[ds]['ac']['em']:.1f} / {summary[ds]['ac']['tok']:,.0f} "
                                            f"({summary[ds]['ac']['samples']:.1f} samples)" if "ac" in summary[ds] else "—"
                                            for ds in summary) + " |")
    (base / "summary.json").write_text(json.dumps({"summary": summary, "sc_curve": curves}, indent=1))
    (base / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
