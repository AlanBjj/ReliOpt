"""[E8] Summary of the execution-layer ablations: EM / F1 / tokens per variant, and the paired bootstrap of each variant
against the full execution layer (10,000 resamples over questions, 95% interval).
Input:  results/e8/<tag>/<ds>/<variant>.jsonl (scripts/e8_exec_ablation.py); with --split cal, results/e8/<tag>/cal/<ds>/
Output: results/e8/<tag>[/<split>]/summary.json and summary.md
Usage:  python3 scripts/e8_summarize.py --tag llama31-8b [--split cal]      standard library only
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["hotpotqa", "2wikimultihopqa", "musique"]
VARIANTS = ["full", "no_sink", "short", "k5", "cmp_llm", "untyped", "no_check"]


def load(path):
    return {r["qid"]: r for r in map(json.loads, path.open())} if path.exists() else {}


def boot(a, b, n=10000, seed=0):
    rng = random.Random(seed)
    qs = sorted(set(a) & set(b))
    d = [a[q]["em"] - b[q]["em"] for q in qs]
    N = len(d)
    mean = 100 * sum(d) / N
    bs = sorted(100 * sum(d[rng.randrange(N)] for _ in range(N)) / N for _ in range(n))
    return mean, bs[int(0.025 * n)], bs[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="test", choices=["test", "tune", "cal"])
    args = ap.parse_args()
    base = ROOT / "results" / "e8" / args.tag
    if args.split != "test":
        base = base / args.split
    summary = {}
    md = [f"# e8 execution-layer ablations ({args.tag}, {args.split}, no verification)", "",
          "ΔEM = EM(variant) - EM(full), paired bootstrap 95% interval over questions (10,000 resamples).", "",
          "| Dataset | Variant | EM | F1 | Tokens/q | ΔEM | 95% CI | Errors | Fallback plans |", "|---|---|---|---|---|---|---|---|---|"]
    for ds in DATASETS:
        rows = {v: load(base / ds / f"{v}.jsonl") for v in VARIANTS}
        full = rows["full"]
        summary[ds] = {}
        for v in VARIANTS:
            r = rows[v]
            if not r:
                continue
            n = len(r)
            s = {"n": n, "em": 100 * sum(x["em"] for x in r.values()) / n, "f1": 100 * sum(x["f1"] for x in r.values()) / n,
                 "tok": sum(x["tok"] for x in r.values()) / n, "errors": sum("error" in x for x in r.values()),
                 "fallback": sum(bool(x.get("fallback")) for x in r.values())}
            if v != "full" and full:
                s["d_em"], s["lo"], s["hi"] = boot(r, full)
            summary[ds][v] = s
            ci = f"[{s['lo']:+.1f}, {s['hi']:+.1f}]" if "lo" in s else "—"
            de = f"{s['d_em']:+.1f}" if "d_em" in s else "—"
            md.append(f"| {ds} | {v} | {s['em']:.1f} | {s['f1']:.1f} | {s['tok']:,.0f} | {de} | {ci} | {s['errors']} | {s['fallback']} |")
    base.mkdir(parents=True, exist_ok=True)
    (base / "summary.json").write_text(json.dumps(summary, indent=1))
    (base / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
