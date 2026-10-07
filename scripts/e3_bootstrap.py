"""Paired bootstrap of the main comparisons: the method at each priced budget against every external
baseline, per dataset, as the EM difference with a 95% percentile interval over 10,000 resamples of the test questions.

Per question, a sampled method's EM is its mean over seeds (so the interval reflects question sampling, not seed noise).
Input:  results/e2/<tag>/test/<ds>/*.jsonl, results/e3/<tag>/test/<ds>/runs/full_s*.jsonl, results/e3/<tag>/prices.json
Output: results/e3/<tag>/test/bootstrap.{json,md}
Usage:  python scripts/e3_bootstrap.py --tag llama31-8b      standard library only
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
B = 10000


def per_question(files, get):
    """qid -> mean over files (seeds) of get(row)."""
    acc = defaultdict(list)
    for p in files:
        for l in open(p):
            r = json.loads(l)
            v = get(r)
            if v is not None:
                acc[r["qid"]].append(v)
    return {q: sum(v) / len(v) for q, v in acc.items()}


def baselines(ds_dir):
    out = {}
    simple = sorted(ds_dir.glob("simple_s*.jsonl"))
    if simple:
        out["rag"] = per_question(simple[:1], lambda r: r["rag"]["em"])
        out["sc5"] = per_question(simple, lambda r: r["sc5"]["em"])
    for m in ("selfask", "ircot", "searcho1", "pyrag", "planbudget", "cove"):
        files = [p for p in sorted(ds_dir.glob(f"{m}*.jsonl")) if p.stem == m or p.stem.startswith(m + "_")]
        if files:
            out[m] = per_question(files, lambda r: r.get("em", 0.0))
    return out


def boot(a, b, rng):
    qs = sorted(set(a) & set(b))
    d = [a[q] - b[q] for q in qs]
    n = len(d)
    stats = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return {"n": n, "diff": 100 * sum(d) / n, "lo": 100 * stats[int(0.025 * B)], "hi": 100 * stats[int(0.975 * B) - 1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    base = ROOT / "results" / "e3" / args.tag
    prices = json.loads((base / "prices.json").read_text())
    rng = random.Random(0)
    res = {}
    lines = [f"# Paired bootstrap ({args.tag}, test): EM(ours) - EM(baseline), 95% interval, {B} resamples", "",
             "| Dataset | Budget | Baseline | n | ΔEM | 95% CI |", "|---|---|---|---|---|---|"]
    for ds in DATASETS:
        runs = sorted((base / "test" / ds / "runs").glob("full_s*.jsonl"))
        if not runs or ds not in prices:
            continue
        bl = baselines(ROOT / "results" / "e2" / args.tag / "test" / ds)
        res[ds] = {}
        for name, lam in prices[ds].items():
            ours = per_question(runs, lambda r, lam=lam: r.get("em", 0.0) if abs(r.get("lam", -1) - lam) <= 1e-12 * max(1, lam)
                                else None)
            res[ds][name] = {}
            for m, b in bl.items():
                r = boot(ours, b, rng)
                res[ds][name][m] = r
                lines.append(f"| {ds} | {name} | {m} | {r['n']} | {r['diff']:+.1f} | [{r['lo']:+.1f}, {r['hi']:+.1f}] |")
    (base / "test" / "bootstrap.json").write_text(json.dumps(res, indent=1))
    (base / "test" / "bootstrap.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
