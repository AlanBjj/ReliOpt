"""e1 side check: where do the simple reference methods (CoT, Standard RAG, Self-Consistency) land on the pilot questions?

Gives an early estimate of how the plan-based pipeline compares with the first batch of baselines on the same 200 + 200
calibration questions. The proper baseline runs are e2, on the test subsets.
Input:  results/e1/ids/<ds>.txt, cache/bm25/<ds>/, vLLM servers (ports 8100 + GPU id)
Output: results/e1/<run>/baselines_<ds>.jsonl (one row per question), printed means
Usage:  python scripts/e1_baselines.py --tag llama31-8b --gpus 0,1,2,3 --run r2
Resumable through the LLM cache (cache/llm/<tag>.sqlite): a rerun repeats no finished call and rewrites the output.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reliopt.baselines.simple import run_simple  # noqa: E402
from reliopt.data.ircot import load_questions  # noqa: E402
from reliopt.llm.client import LLM, run_parallel  # noqa: E402
from reliopt.retrieval.bm25 import BM25Index  # noqa: E402
from reliopt.utils import load_env  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--datasets", nargs="+", default=["2wikimultihopqa", "musique"])
    ap.add_argument("--run", default="r2")
    ap.add_argument("--workers", type=int, default=160)
    args = ap.parse_args()
    load_env()
    eps = [f"http://127.0.0.1:{8100 + int(g)}" for g in args.gpus.split(",")]
    cache = ROOT / "cache" / "llm" / f"{args.tag}.sqlite"
    cache.parent.mkdir(parents=True, exist_ok=True)
    llm = LLM(args.tag, eps, cache_path=cache)
    for ds in args.datasets:
        ids = (ROOT / "results" / "e1" / "ids" / f"{ds}.txt").read_text().split()
        qs = load_questions(ds, "train", qids=ids)
        index = BM25Index(ds)
        rows = run_parallel(lambda q: {"qid": q["qid"], **run_simple(llm, index, q)}, qs, workers=args.workers)
        out = ROOT / "results" / "e1" / args.run / f"baselines_{ds}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        for m in ("cot", "rag", "sc3", "sc5", "sc10"):
            em = 100 * sum(r[m]["em"] for r in rows) / len(rows)
            f1 = 100 * sum(r[m]["f1"] for r in rows) / len(rows)
            tok = sum(r[m]["tok"] for r in rows) / len(rows)
            print(f"{ds:16s} {m:5s} EM {em:5.1f}  F1 {f1:5.1f}  tokens/question {tok:7.0f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
