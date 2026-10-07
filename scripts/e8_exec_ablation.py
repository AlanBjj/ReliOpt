"""[E8] Execution-layer ablations: the typed plan executed with no verification (Plan-Only), one design choice removed at a time.

Why: the repositioned paper claims the typed query plan and its typed physical execution as a contribution,
so each part of the execution layer needs test-set evidence. Every variant runs the plan with NO operator (greedy, one run),
on the same backbone, retriever and test questions as e3; "full" is the method's own configuration
(results/e3/<tag>/config.json) and reproduces e3's Plan-Only through the shared LLM cache.
  full      the method's execution layer (TXT reader copies the answer span, reads k passages, the sink also reads the question)
  no_sink   the sink TXT step does not see the original question or its passages          (sink_q=False)
  short     the TXT reader writes a short answer instead of copying the passage's words  (txt="short")
  k5        the TXT reader reads 5 passages instead of 8                                 (k=5)
  cmp_llm   CMP steps are answered by the model directly instead of by a program        (cmp_direct=True)
  untyped   every step is executed as a retrieve-and-read look-up, whatever its type     (untyped=True)
  no_check  the planner keeps the plans the two round-4 rules reject or rewrite          (strict=False)
Input:  results/e0/ids/<ds>_test.txt, results/e3/<tag>/config.json, cache/bm25/<ds>/; a running vLLM server for <tag>
Output: results/e8/<tag>/<ds>/<variant>.jsonl (one row per question: pred, em, f1, tok, plan facts); with --split cal or
        tune, results/e8/<tag>/<split>/<ds>/<variant>.jsonl (the 500 calibration questions are the training questions of e3,
        whose plan-only calls are cached, so "full" there is free)
Usage:  python scripts/e8_exec_ablation.py --tag llama31-8b --gpus 2,4,6 [--datasets ...] [--variants ...]
        a finished file is skipped; an interrupted variant resumes from the LLM cache
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reliopt.data.ircot import load_questions  # noqa: E402
from reliopt.exec.replay import NO_VERIFY, Pipeline  # noqa: E402
from reliopt.llm.client import LLM, run_parallel  # noqa: E402
from reliopt.retrieval.bm25 import BM25Index  # noqa: E402
from reliopt.utils import load_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"test": "test_subsampled", "tune": "dev_subsampled", "cal": "train"}   # cal: route-2 quick check (scripts/e8_select.py)
VARIANTS = {"full": {}, "no_sink": {"sink_q": False}, "short": {"txt": "short"}, "k5": {"k": 5},
            "cmp_llm": {"cmp_direct": True}, "untyped": {"untyped": True}, "no_check": {"strict": False}}
INFRA_ERRORS = ("LLM request failed", "database is locked")
SEED = 17   # greedy throughout; the seed only names the (unused) sample pool


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--split", choices=list(SPLITS), default="test")
    ap.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimultihopqa", "musique"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--workers", type=int, default=160)
    args = ap.parse_args()
    load_env()
    cfg = json.loads((ROOT / "results" / "e3" / args.tag / "config.json").read_text())
    base = {"k": cfg["k"], "txt": cfg["txt"], "sink_q": cfg.get("sink_q", False)}
    eps = [f"http://127.0.0.1:{8100 + int(g)}" for g in args.gpus.split(",")]
    llm = LLM(args.tag, eps, cache_path=ROOT / "cache" / "llm" / f"{args.tag}.sqlite", timeout=3600)
    for ds in args.datasets:
        ids = (ROOT / "results" / "e0" / "ids" / f"{ds}_{args.split}.txt").read_text().split()
        qs = load_questions(ds, SPLITS[args.split], qids=ids)
        assert len(qs) == len(ids), (ds, len(qs), len(ids))
        index = None
        for v in args.variants:
            sub = ROOT / "results" / "e8" / args.tag if args.split == "test" else ROOT / "results" / "e8" / args.tag / args.split
            out = sub / ds / f"{v}.jsonl"
            if out.exists():
                print(f"{ds:16s} {v:9s} done, skipped")
                continue
            index = index or BM25Index(ds)
            pipe = Pipeline(llm, index, seed=SEED, **{**base, **VARIANTS[v]})
            t0 = time.time()

            def one(q, pipe=pipe):
                try:
                    plan = pipe.plan(q["question"])
                    r = pipe.run(q, plan, NO_VERIFY)
                    return {"qid": q["qid"], "pred": r["pred"], "em": r["em"], "f1": r["f1"], "acc": r.get("acc"),
                            "tok": sum(r["tok"].values()), "tok_parts": r["tok"], "calls": sum(r["calls"].values()),
                            "fallback": plan["fallback"], "attempts": plan["attempts"],
                            "kinds": [s["kind"] for s in plan["steps"]]}
                except Exception:
                    return {"qid": q["qid"], "pred": "", "em": 0.0, "f1": 0.0, "tok": 0, "calls": 0,
                            "error": traceback.format_exc()[-800:]}
            rows = run_parallel(one, qs, workers=args.workers)
            infra = sum(1 for r in rows if any(m in r.get("error", "") for m in INFRA_ERRORS))
            if infra:   # server or cache failures, not the variant's: leave the file unwritten so a rerun redoes them
                print(f"{ds:16s} {v:9s} {infra} questions hit infrastructure errors; not written, rerun to finish")
                sys.stdout.flush()
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            tmp.rename(out)
            n = len(rows)
            print(f"{ds:16s} {v:9s} EM {100 * sum(r['em'] for r in rows) / n:5.1f}  "
                  f"F1 {100 * sum(r['f1'] for r in rows) / n:5.1f}  tokens/question {sum(r['tok'] for r in rows) / n:6.0f}  "
                  f"errors {sum('error' in r for r in rows)}  {time.time() - t0:.0f}s")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
