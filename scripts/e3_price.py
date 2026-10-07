"""e3 step 4: price the main-table budgets: bisect lambda on the tuning questions so that the method's
mean verification tokens per question meet each target budget, then the test run uses those lambdas unchanged.

Targets per dataset: B_SC = the measured mean tokens per question of Self-Consistency (k = 5) on that dataset's test
questions (results/e2/<tag>/test/summary.json), and B_SC / 2. They are budgets for the verification tokens (the method's
plan-only cost is common to every allocation and reported separately).
Input:  results/e3/<tag>/config.json, models/<ds>_logreg.pkl, results/e2/<tag>/test/summary.json; vLLM servers
Output: results/e3/<tag>/prices.json  {ds: {"B_SC": lambda, "B_SC/2": lambda}} and prices_detail.json (targets, achieved)
Usage:  python scripts/e3_price.py --tag llama31-8b --gpus 0,1,2,3 [--datasets ...]
"""
import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reliopt.data.ircot import load_questions  # noqa: E402
from reliopt.exec.adaptive import Allocator, run_adaptive  # noqa: E402
from reliopt.exec.replay import Pipeline  # noqa: E402
from reliopt.llm.client import LLM, run_parallel  # noqa: E402
from reliopt.retrieval.bm25 import BM25Index  # noqa: E402
from reliopt.utils import load_env  # noqa: E402

warnings.filterwarnings("ignore")
LO, HI, ITERS, SEED = -9.0, -2.0, 14, 17


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimultihopqa", "musique"])
    ap.add_argument("--workers", type=int, default=100)
    args = ap.parse_args()
    load_env()
    base = ROOT / "results" / "e3" / args.tag
    cfg = json.loads((base / "config.json").read_text())
    sc = json.loads((ROOT / "results" / "e2" / args.tag / "test" / "summary.json").read_text())["summary"]
    llm = LLM(args.tag, [f"http://127.0.0.1:{8100 + int(g)}" for g in args.gpus.split(",")],
              cache_path=ROOT / "cache" / "llm" / f"{args.tag}.sqlite")
    prices = json.loads((base / "prices.json").read_text()) if (base / "prices.json").exists() else {}
    detail = json.loads((base / "prices_detail.json").read_text()) if (base / "prices_detail.json").exists() else {}
    for ds in args.datasets:
        qm = pickle.load(open(base / "models" / f"{ds}_logreg.pkl", "rb"))
        ids = (ROOT / "results" / "e0" / "ids" / f"{ds}_tune.txt").read_text().split()
        qs = load_questions(ds, "dev_subsampled", qids=ids)
        pipe = Pipeline(llm, BM25Index(ds), seed=SEED, k=cfg["k"], txt=cfg["txt"], sink_q=cfg.get("sink_q", False))
        plans = {q["qid"]: pipe.plan(q["question"]) for q in qs}

        def spend(lam):
            rows = run_parallel(lambda q: run_adaptive(pipe, q, plans[q["qid"]], Allocator(qm, lam)), qs,
                                workers=args.workers)
            return sum(r["tok"]["verify"] for r in rows) / len(rows), 100 * sum(r["em"] for r in rows) / len(rows)
        targets = {"B_SC": sc[ds]["sc5"]["tok"], "B_SC/2": sc[ds]["sc5"]["tok"] / 2}
        prices[ds], detail[ds] = {}, {}
        top, _ = spend(0.0)
        for name, target in targets.items():
            if target >= top:   # even lambda = 0 spends less than the budget
                lam, (tok, em) = 0.0, spend(0.0)
            else:
                lo, hi = LO, HI   # spend(10**lo) >= target > spend(10**hi)
                for _ in range(ITERS):
                    mid = (lo + hi) / 2
                    if spend(10 ** mid)[0] > target:
                        lo = mid
                    else:
                        hi = mid
                lam = 10 ** hi
                tok, em = spend(lam)
            prices[ds][name] = lam
            detail[ds][name] = {"target_verify_tok": round(target, 1), "lambda": lam, "tune_verify_tok": round(tok, 1),
                                "tune_em": round(em, 2), "lambda0_verify_tok": round(top, 1)}
            print(f"{ds} {name}: target {target:.0f} verify tok/question -> lambda {lam:.3g} "
                  f"(tune: {tok:.0f} tok, EM {em:.1f}; lambda=0 spends {top:.0f})"); sys.stdout.flush()
        (base / "prices.json").write_text(json.dumps(prices, indent=1))
        (base / "prices_detail.json").write_text(json.dumps(detail, indent=1))


if __name__ == "__main__":
    main()
