"""e2: the external baselines of the main table on the IRCoT test subsets (or the 100 tuning questions).

Methods:
  simple    CoT, Standard RAG and a 10-sample Self-Consistency pool per seed (SC-k and Adaptive-Consistency are replayed
            from the pool offline by scripts/e2_summarize.py)
  selfask   Self-Ask (greedy, one run)
  cove      Chain-of-Verification, factored (greedy, one run; appendix table)
  ircot_budget  IRCoT at reduced budgets for its quality-cost curve: (demonstrations, max paragraphs) in IRCOT_BUDGETS,
            the tuned K (official IRCoT keeps every demonstration that fits 8000 tokens and up to 15 paragraphs)
  ircot     IRCoT (greedy, one run) with K paragraphs per retrieval; --ircot-k auto takes, per dataset, the K in {4, 6, 8}
            with the best EM on the tuning split (ties: F1, then the smaller K), so the tuning split must be run first
  searcho1  Search-o1 (sampled, one run per seed)
  pyrag     PyRAG (sampled, one run per seed)
  planbudget Plan-and-Budget (sampled, one run per seed) with its BM25 Reference built from the question ("question") or
            also from each sub-question ("subq"); --pb-retrieval auto takes the better one on the tuning split (ties:
            "question"); on the tuning split only the first seed is run
Input:  results/e0/ids/<ds>_{test,tune}.txt, cache/bm25/<ds>/, cache/ircot/ircot_repo/, vLLM servers (ports 8100 + GPU id)
Output: results/e2/<tag>/<split>/<ds>/<method>[_k<K>][_s<seed>].jsonl, one row per question
        --split cal runs on the 500 calibration (training) questions of e3 (2026-10-07, e9: what escalating a question to
        IRCoT / Search-o1 gains, estimated without test labels; silver labels for the Adaptive-RAG classifier)
Usage:  python scripts/e2_baselines.py --tag llama31-8b --gpus 0,1,2,3 --split test --datasets musique \
            --methods simple selfask ircot searcho1 pyrag planbudget [--seeds 17 29 43] [--ircot-k auto] [--pb-retrieval auto]
Resumable: a finished output file is skipped; a half-done one is redone, mostly from the LLM cache (cache/llm/<tag>.sqlite).
A run in which some question hit a server-side request failure is not written (the next run redoes it from the cache).
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reliopt.baselines.cove import CoVe  # noqa: E402
from reliopt.baselines.ircot import IRCoT  # noqa: E402
from reliopt.baselines.planbudget import PlanAndBudget  # noqa: E402
from reliopt.baselines.pyrag import PyRAG  # noqa: E402
from reliopt.baselines.searcho1 import SearchO1  # noqa: E402
from reliopt.baselines.selfask import SelfAsk  # noqa: E402
from reliopt.baselines.simple import run_simple  # noqa: E402
from reliopt.data.ircot import load_questions  # noqa: E402
from reliopt.eval.metrics import score  # noqa: E402
from reliopt.llm.client import LLM, run_parallel  # noqa: E402
from reliopt.retrieval.bm25 import BM25Index  # noqa: E402
from reliopt.utils import load_env  # noqa: E402

SPLITS = {"test": "test_subsampled", "tune": "dev_subsampled", "cal": "train", "devcal": "dev", "devcal2": "dev"}   # cal: e9 (Adaptive-RAG labels); devcal: e9 escalation maps (scripts/e9_devcal.py)
IRCOT_KS = (4, 6, 8)
IRCOT_BUDGETS = ((1, 6), (2, 10), (4, 15))   # (demonstrations, max accumulated paragraphs) of the reduced-budget IRCoT
PB_RETRIEVAL = ("question", "subq")
THINKING_SOLVER = None   # set for Qwen3 in main()


def out_dir(tag, split, ds):
    return ROOT / "results" / "e2" / tag / split / ds


def pick_on_tune(tag, ds, stems, hint):
    """The stem with the best EM on the tuning split (ties: F1, then the earlier stem in `stems`)."""
    best = None
    for rank, stem in enumerate(stems):
        p = out_dir(tag, "tune", ds) / f"{stem}.jsonl"
        if not p.exists():
            raise SystemExit(f"auto selection needs {p}; first run {hint}")
        rows = [json.loads(l) for l in p.open()]
        key = (sum(r["em"] for r in rows) / len(rows), sum(r["f1"] for r in rows) / len(rows), -rank)
        best = max(best, (key, stem)) if best else (key, stem)
    return best[1]


def runs(method, args, ds):
    """(file stem, per-question function) for every run of `method` on dataset `ds`."""
    if method == "simple":
        return [(f"simple_s{s}", lambda llm, idx, q, s=s: run_simple(llm, idx, q, seed=s)) for s in args.seeds]
    if method == "selfask":
        return [("selfask", lambda llm, idx, q: SelfAsk(llm, idx).run(q))]
    if method == "cove":
        return [("cove", lambda llm, idx, q: CoVe(llm, idx).run(q))]
    if method == "ircot":
        if args.ircot_k == ["auto"]:
            best = pick_on_tune(args.tag, ds, [f"ircot_k{k}" for k in IRCOT_KS], "--split tune --methods ircot --ircot-k 4 6 8")
            ks = [int(best[len("ircot_k"):])]
        else:
            ks = [int(k) for k in args.ircot_k]
        return [(f"ircot_k{k}", lambda llm, idx, q, k=k: IRCoT(llm, idx, ds, k=k).run(q)) for k in ks]
    if method == "ircot_budget":   # budget-reduced IRCoT (fewer demonstrations, fewer paragraphs), tuned K
        best = pick_on_tune(args.tag, ds, [f"ircot_k{k}" for k in IRCOT_KS], "--split tune --methods ircot --ircot-k 4 6 8")
        k = int(best[len("ircot_k"):])
        return [(f"ircotb_d{d}_p{p}_k{k}", lambda llm, idx, q, d=d, p=p: IRCoT(llm, idx, ds, k=k, n_demos=d, max_paras=p).run(q))
                for d, p in IRCOT_BUDGETS]
    if method == "searcho1":
        return [(f"searcho1_s{s}", lambda llm, idx, q, s=s: SearchO1(llm, idx, seed=s).run(q)) for s in args.seeds]
    if method == "pyrag":
        return [(f"pyrag_s{s}", lambda llm, idx, q, s=s: PyRAG(llm, idx, seed=s).run(q)) for s in args.seeds]
    if method == "planbudget":
        seeds = args.seeds if args.split == "test" else args.seeds[:1]
        if args.pb_retrieval == ["auto"]:
            best = pick_on_tune(args.tag, ds, [f"planbudget_{r}_s{seeds[0]}" for r in PB_RETRIEVAL],
                                "--split tune --methods planbudget --pb-retrieval question subq")
            modes = [best.split("_")[1]]
        else:
            modes = args.pb_retrieval
        return [(f"planbudget_{r}_s{s}", lambda llm, idx, q, r=r, s=s: PlanAndBudget(llm, idx, seed=s, retrieval=r,
                                                                                    solver=THINKING_SOLVER).run(q))
                for r in modes for s in seeds]
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--split", choices=list(SPLITS), default="test")
    ap.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimultihopqa", "musique"])
    ap.add_argument("--methods", nargs="+", default=["simple", "selfask", "ircot", "searcho1", "pyrag"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    ap.add_argument("--ircot-k", nargs="+", default=["auto"])
    ap.add_argument("--pb-retrieval", nargs="+", default=["auto"])
    ap.add_argument("--workers", type=int, default=160)
    args = ap.parse_args()
    load_env()
    eps = [f"http://127.0.0.1:{8100 + int(g)}" for g in args.gpus.split(",")]
    cache = ROOT / "cache" / "llm" / f"{args.tag}.sqlite"
    cache.parent.mkdir(parents=True, exist_ok=True)
    # long generations (Plan-and-Budget allows 8192 tokens) can wait a long time in a full vLLM queue
    llm = LLM(args.tag, eps, cache_path=cache, timeout=3600)
    global THINKING_SOLVER   # Qwen3: Plan-and-Budget's solving call runs in thinking mode
    if args.tag == "qwen3-8b":
        THINKING_SOLVER = LLM("qwen3-8b-think", eps, cache_path=cache, timeout=3600)
    for ds in args.datasets:
        ids = (ROOT / "results" / "e0" / "ids" / f"{ds}_{args.split}.txt").read_text().split()
        qs = load_questions(ds, SPLITS[args.split], qids=ids)
        assert len(qs) == len(ids), (ds, len(qs), len(ids))
        index = None
        for m in args.methods:
            for stem, fn in runs(m, args, ds):
                out = out_dir(args.tag, args.split, ds) / f"{stem}.jsonl"
                if out.exists():
                    print(f"{ds:16s} {stem:14s} done, skipped")
                    continue
                index = index or BM25Index(ds)
                t0 = time.time()

                def one(q, fn=fn, simple=(m == "simple")):
                    try:
                        r = fn(llm, index, q)
                    except Exception:
                        if simple:
                            raise
                        r = {"pred": "", "tok": 0, "calls": 0, "error": traceback.format_exc()[-800:]}
                    return {"qid": q["qid"], **r} if simple else {"qid": q["qid"], **r, **score(r["pred"], q["answers"])}
                rows = run_parallel(one, qs, workers=args.workers)
                infra = sum(1 for r in rows if any(m in r.get("error", "") for m in ("LLM request failed",
                                                                                      "database is locked")))
                if infra:   # server-side failures, not the method's: leave the file unwritten so a rerun redoes them
                    print(f"{ds:16s} {stem:14s} {infra} questions hit LLM request failures; not written, rerun to finish")
                    sys.stdout.flush()
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(".tmp")
                tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
                tmp.rename(out)
                n = len(rows)
                for name, get in ([("cot", lambda r: r["cot"]), ("rag", lambda r: r["rag"]), ("sc5", lambda r: r["sc5"]),
                                   ("sc10", lambda r: r["sc10"])] if m == "simple" else [(stem, lambda r: r)]):
                    em = 100 * sum(get(r)["em"] for r in rows) / n
                    f1 = 100 * sum(get(r)["f1"] for r in rows) / n
                    tok = sum(get(r)["tok"] for r in rows) / n
                    errs = sum(1 for r in rows if "error" in r)
                    print(f"{ds:16s} {name:14s} EM {em:5.1f}  F1 {f1:5.1f}  tokens/question {tok:7.0f}  errors {errs}"
                          f"  {time.time() - t0:.0f}s")
                sys.stdout.flush()


if __name__ == "__main__":
    main()
