"""e1 side check: the strong multi-step baselines (IRCoT, Self-Ask, PyRAG, Search-o1) on the pilot questions.

Same backbone, same BM25 corpora and the same 200 + 200 calibration questions as the e1 rounds, so their EM / tokens can be
set against the plan-based pipeline before the full e2-e5 runs on the test subsets.
Input:  results/e1/ids/<ds>.txt, cache/bm25/<ds>/, cache/ircot/ircot_repo/ (prompts), vLLM servers (ports 8100 + GPU id)
Output: results/e1/<run>/strong_<method>_<ds>.jsonl (one row per question), printed means
Usage:  python scripts/e1_strong.py --tag llama31-8b --gpus 0,1,2,3 --run r3 [--methods ircot selfask pyrag searcho1]
Resumable through the LLM cache (cache/llm/<tag>.sqlite); each (method, dataset) file is rewritten when it is rerun.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reliopt.baselines.ircot import IRCoT  # noqa: E402
from reliopt.baselines.pyrag import PyRAG  # noqa: E402
from reliopt.baselines.searcho1 import SearchO1  # noqa: E402
from reliopt.baselines.selfask import SelfAsk  # noqa: E402
from reliopt.data.ircot import load_questions  # noqa: E402
from reliopt.eval.metrics import score  # noqa: E402
from reliopt.llm.client import LLM, run_parallel  # noqa: E402
from reliopt.retrieval.bm25 import BM25Index  # noqa: E402
from reliopt.utils import load_env  # noqa: E402

METHODS = {"ircot": lambda llm, idx, ds: IRCoT(llm, idx, ds), "selfask": lambda llm, idx, ds: SelfAsk(llm, idx),
           "pyrag": lambda llm, idx, ds: PyRAG(llm, idx), "searcho1": lambda llm, idx, ds: SearchO1(llm, idx)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--datasets", nargs="+", default=["2wikimultihopqa", "musique"])
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--run", default="r3")
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
        for m in args.methods:
            t0 = time.time()

            def one(q, m=m):
                try:   # a fresh method object per question: the methods keep per-question token counters
                    r = METHODS[m](llm, index, ds).run(q)
                except Exception:
                    r = {"pred": "", "tok": 0, "calls": 0, "error": traceback.format_exc()[-800:]}
                return {"qid": q["qid"], **r, **score(r["pred"], q["answers"])}
            rows = run_parallel(one, qs, workers=args.workers)
            out = ROOT / "results" / "e1" / args.run / f"strong_{m}_{ds}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            n = len(rows)
            errs = sum(1 for r in rows if "error" in r)
            print(f"{ds:16s} {m:9s} EM {100 * sum(r['em'] for r in rows) / n:5.1f}  F1 {100 * sum(r['f1'] for r in rows) / n:5.1f}"
                  f"  tokens/question {sum(r['tok'] for r in rows) / n:7.0f}  calls/question {sum(r['calls'] for r in rows) / n:5.1f}"
                  f"  errors {errs}  {time.time() - t0:.0f}s")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
