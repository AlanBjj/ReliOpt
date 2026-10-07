"""Paired bootstrap of the method at B_SC against the internal variants (Plan-Only, Verify-All, Type-Static, Question-Level,
ablations), per question EM averaged over seeds, 10,000 resamples, 95% interval.

The method's point is its priced lambda for B_SC. A variant without a priced point is taken at its operating point (lambda,
or policy for Type-Static) whose mean verification tokens per question is closest to the method's; Verify-All has one point.
Question-Level chooses its policy per question, so its points are its lambdas.
Input:  results/e3/<tag>/test/<ds>/runs/<variant>_s<seed>.jsonl, results/e3/<tag>/prices.json
Output: results/e3/<tag>/test/bootstrap_internal.json and .md
Usage:  python3 scripts/e3_bootstrap_internal.py --tag llama31-8b      standard library only
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["hotpotqa", "2wikimultihopqa", "musique"]
VARIANTS = ["plan_only", "verify_all", "type_static", "question_level", "no_prop", "prior_eps", "no_probe", "gbdt"]


def load(base, ds, v):
    rows = []
    for p in sorted((base / "test" / ds / "runs").glob(f"{v}_s*.jsonl")):
        rows += [json.loads(line) for line in p.open()]
    return rows


def points(rows, per_question_policy):
    pts = defaultdict(lambda: defaultdict(list))
    for r in rows:
        pts[r.get("lam"), None if per_question_policy else r.get("policy")][r["qid"]].append(r)
    return pts


def em(per_q):
    return {q: sum(x["em"] for x in rs) / len(rs) for q, rs in per_q.items()}


def vtok(per_q):
    xs = [x["tok"]["verify"] for rs in per_q.values() for x in rs]
    return sum(xs) / len(xs)


def boot(a, b, rng, n=10000):
    qs = sorted(a)
    d = [a[q] - b[q] for q in qs]
    N = len(d)
    bs = sorted(100 * sum(d[rng.randrange(N)] for _ in range(N)) / N for _ in range(n))
    return 100 * sum(d) / N, bs[int(0.025 * n)], bs[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    base = ROOT / "results" / "e3" / args.tag
    prices = json.loads((base / "prices.json").read_text())
    rng = random.Random(0)
    out = {}
    md = [f"# Paired bootstrap against internal variants ({args.tag}, test, B_SC)", "",
          "ΔEM = EM(method) - EM(variant), per question averaged over seeds; 95% interval, 10,000 resamples.", "",
          "| Dataset | Variant | Verify tok (variant) | EM (variant) | EM (method) | ΔEM | 95% CI |", "|---|---|---|---|---|---|---|"]
    for ds in DATASETS:
        full = points(load(base, ds, "full"), False)
        lam = prices[ds]["B_SC"]
        key = min(full, key=lambda k: abs((k[0] or 0) - lam))
        o = em(full[key])
        out[ds] = {"method": {"lam": key[0], "em": 100 * sum(o.values()) / len(o), "verify_tok": vtok(full[key])}}
        for v in VARIANTS:
            pts = points(load(base, ds, v), v == "question_level")
            if not pts:
                continue
            k = min(pts, key=lambda k: abs(vtok(pts[k]) - out[ds]["method"]["verify_tok"]))
            b = em(pts[k])
            d, lo, hi = boot(o, b, rng)
            out[ds][v] = {"point": list(k), "verify_tok": vtok(pts[k]), "em": 100 * sum(b.values()) / len(b),
                          "d_em": d, "lo": lo, "hi": hi}
            md.append(f"| {ds} | {v} | {vtok(pts[k]):,.0f} | {out[ds][v]['em']:.1f} | {out[ds]['method']['em']:.1f} | "
                      f"{d:+.1f} | [{lo:+.1f}, {hi:+.1f}]{' *' if lo > 0 or hi < 0 else ''} |")
    (base / "test" / "bootstrap_internal.json").write_text(json.dumps(out, indent=1))
    (base / "test" / "bootstrap_internal.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
