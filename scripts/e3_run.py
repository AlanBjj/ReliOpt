"""e3 step 3: run the method and the internal comparisons of Table 2 on the tuning or test questions.

Variants, each with seeds 17 / 29 / 43 for the sample pool (plans and default executions are greedy):
  full        the method: eps-hat (logistic regression + isotonic), propagation, VOI probing, price lambda
  no_probe    w/o escalation (no PROBE; operators chosen from the free features)
  no_prop     w/o propagation (pi = S = 1)
  prior_eps   w/o instance estimates (eps = type prior)
  gbdt        gradient-boosted trees replace the eps-hat estimator
  question_level  per-question budget from the question-level error estimate, spread evenly over its steps
  plan_only / verify_all (DEEP on every step) / type_static (the calibration frontier policies, see scripts/e3_fit.py)
The price-based variants run at every lambda of the grid plus, when results/e3/<tag>/prices.json exists, the prices
bisected on the tuning split for the main-table budgets (scripts/e3_price.py).
Input:  results/e3/<tag>/config.json, models/<ds>_<estimator>.pkl, models/<ds>_summary.json; vLLM servers
Output: results/e3/<tag>/<split>/<ds>/runs/<variant>_s<seed>.jsonl, one row per (question, lambda or policy)
Usage:  python scripts/e3_run.py --tag llama31-8b --gpus 0,1,2,3 --split test [--datasets ...] [--variants ...]
Resumable: a finished output file is skipped; calls are cached, so a rerun of a half-done file repeats no model call.
Sharding (--shard k --nshards N): process k handles questions k, k+N, ...; it writes runs/shards/<file>.<k>of<N>.jsonl
and the shard that completes a set merges it into the final file (in question order). Shards never share a question, so
parallel processes never race on the same cache key (the work is CPU-bound in Python, one process per core helps).
"""
import argparse
import json
import os
import pickle
import sys
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reliopt.data.ircot import load_questions  # noqa: E402
from reliopt.exec.adaptive import Allocator, run_adaptive, run_question_level  # noqa: E402
from reliopt.exec.replay import NO_VERIFY, Pipeline, static  # noqa: E402
from reliopt.llm.client import LLM, run_parallel  # noqa: E402
from reliopt.retrieval.bm25 import BM25Index  # noqa: E402
from reliopt.utils import load_env  # noqa: E402

warnings.filterwarnings("ignore")
SPLITS = {"cal": "train", "tune": "dev_subsampled", "test": "test_subsampled"}
LAMS = [0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 1e-3]
PRICED = {"full": dict(), "no_probe": dict(probe=False), "no_prop": dict(propagate=False),
          "prior_eps": dict(prior_eps=True), "gbdt": dict()}
VARIANTS = list(PRICED) + ["question_level", "plan_only", "verify_all", "type_static"]
DEEP = {"TXT": "DEEP", "CMP": "DEEP", "LOG": "DEEP"}
INFRA_ERRORS = ("LLM request failed", "database is locked")


def compact(r):
    return {"pred": r["pred"], "em": r["em"], "f1": r["f1"], "tok": r["tok"], "calls": r["calls"]}


def lambdas(tag, ds):
    p = ROOT / "results" / "e3" / tag / "prices.json"
    priced = json.loads(p.read_text()).get(ds, {}) if p.exists() else {}
    return sorted(set(LAMS) | set(priced.values()))


def runner(variant, pipe, qm, models, lams, frontier):
    """Per-question function returning the rows of one variant on one question."""
    def one(q):
        plan = pipe.plan(q["question"])
        rows = []
        if variant == "plan_only":
            rows.append({"policy": "NONE", **compact(pipe.run(q, plan, NO_VERIFY))})
        elif variant == "verify_all":
            rows.append({"policy": "DEEP", **compact(pipe.run(q, plan, static(DEEP)))})
        elif variant == "type_static":
            for name, _, _ in frontier:
                t, c, l = name.split("|")
                rows.append({"policy": name, **compact(pipe.run(q, plan, static({"TXT": t, "CMP": c, "LOG": l})))})
        elif variant == "question_level":
            for lam in lams:
                r = run_question_level(pipe, q, plan, qm, lam)
                rows.append({"lam": lam, "policy": r["policy"], "eps_q": r["eps_q"], **compact(r)})
        else:
            m = models["gbdt"] if variant == "gbdt" else qm
            for lam in lams:
                r = run_adaptive(pipe, q, plan, Allocator(m, lam, **PRICED[variant]))
                row = {"lam": lam, **compact(r)}
                if variant == "full":
                    row["steps"] = r["steps"]
                rows.append(row)
        return [{"qid": q["qid"], **r} for r in rows]
    return one


def merge(final, nshards, order):
    """Concatenate the shard files of `final` (question order) once all of them exist."""
    parts = [final.parent / "shards" / f"{final.stem}.{k}of{nshards}.jsonl" for k in range(nshards)]
    if final.exists() or not all(p.exists() for p in parts):
        return
    rows = [json.loads(l) for p in parts for l in p.open()]
    rows.sort(key=lambda r: order[r["qid"]])   # stable: keeps each question's rows in lambda / policy order
    tmp = final.with_suffix(f".merge{os.getpid()}.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.rename(final)
    print(f"merged {final.name}: {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimultihopqa", "musique"])
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    ap.add_argument("--workers", type=int, default=160)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    load_env()
    base = ROOT / "results" / "e3" / args.tag
    cfg = json.loads((base / "config.json").read_text())
    eps = [f"http://127.0.0.1:{8100 + int(g)}" for g in args.gpus.split(",")]
    llm = LLM(args.tag, eps, cache_path=ROOT / "cache" / "llm" / f"{args.tag}.sqlite")
    for ds in args.datasets:
        models = {e: pickle.load(open(base / "models" / f"{ds}_{e}.pkl", "rb")) for e in ("logreg", "gbdt")}
        frontier = json.loads((base / "models" / f"{ds}_summary.json").read_text())["type_static_frontier"]
        ids = (ROOT / "results" / "e0" / "ids" / f"{ds}_{args.split}.txt").read_text().split()
        qs = load_questions(ds, SPLITS[args.split], qids=ids)
        order = {q["qid"]: i for i, q in enumerate(qs)}
        qs = qs[args.shard::args.nshards]
        index = BM25Index(ds)
        lams = lambdas(args.tag, ds)
        for seed in args.seeds:
            pipe = Pipeline(llm, index, seed=seed, k=cfg["k"], txt=cfg["txt"], sink_q=cfg.get("sink_q", False))
            for v in args.variants:
                if v == "plan_only" and seed != args.seeds[0]:
                    continue   # greedy throughout: identical for every seed
                final = base / args.split / ds / "runs" / f"{v}_s{seed}.jsonl"
                if final.exists():
                    print(f"{ds} {v} s{seed}: done, skipped"); continue
                out = final if args.nshards == 1 else \
                    final.parent / "shards" / f"{final.stem}.{args.shard}of{args.nshards}.jsonl"
                if out.exists():
                    merge(final, args.nshards, order); continue
                fn = runner(v, pipe, models["logreg"], models, lams, frontier)

                def safe(q, fn=fn):
                    try:
                        return fn(q)
                    except Exception:
                        return [{"qid": q["qid"], "error": traceback.format_exc()[-800:]}]
                rows = [r for rs in run_parallel(safe, qs, workers=args.workers) for r in rs]
                infra = sum(1 for r in rows if any(m in r.get("error", "") for m in INFRA_ERRORS))
                if infra:   # server or cache failures, not the method's: leave unwritten so the next run redoes them
                    print(f"{ds} {v} s{seed} shard {args.shard}/{args.nshards}: {infra} infrastructure errors; "
                          f"not written, rerun to finish"); sys.stdout.flush()
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(f".{os.getpid()}.tmp")   # a second set of processes may run the same shard
                tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
                tmp.rename(out)
                errs = sum("error" in r for r in rows)
                print(f"{ds} {v} s{seed} shard {args.shard}/{args.nshards}: {len(rows)} rows, {errs} errors")
                sys.stdout.flush()
                if args.nshards > 1:
                    merge(final, args.nshards, order)


if __name__ == "__main__":
    main()
