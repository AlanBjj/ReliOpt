"""e3 step 1: materialise questions once: plan, plan-only run, step labels and free features, every
operator on every step (on the plan-only inputs), single-step interventions, and on the calibration split also the full
type-level policy grid (for Type-Static).

Calibration questions fit the quality model (scripts/e3_fit.py); test questions are materialised for the diagnostics of
Table 3 and the oracle, never for fitting. The TXT reader configuration (style, k) is the round-4 choice, read from
results/e3/<tag>/config.json so that every e3 script uses the same one.
Input:  results/e0/ids/<ds>_{cal,tune,test}.txt, cache/bm25/<ds>/, vLLM servers (ports 8100 + GPU id), config.json
Output: results/e3/<tag>/<split>/<ds>/questions.jsonl (rows as in scripts/e1_pilot.py)
Usage:  python scripts/e3_materialize.py --tag llama31-8b --gpus 0,1,2,3 --split cal [--datasets ...]
Resumable: finished questions are kept in questions.partial.jsonl and skipped; model calls are cached.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from e1_pilot import OP_SEED, materialize  # noqa: E402
from reliopt.exec.replay import Pipeline  # noqa: E402
from reliopt.llm.client import LLM  # noqa: E402
from reliopt.retrieval.bm25 import BM25Index  # noqa: E402
from reliopt.utils import load_env  # noqa: E402

SPLITS = {"cal": "train", "tune": "dev_subsampled", "test": "test_subsampled"}


def config(tag):
    p = ROOT / "results" / "e3" / tag / "config.json"
    if not p.exists():
        raise SystemExit(f"{p} missing: write the round-4 TXT choice there first, e.g. {{\"txt\": \"span\", \"k\": 8, \"sink_q\": true}}")
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimultihopqa", "musique"])
    ap.add_argument("--workers", type=int, default=160)
    args = ap.parse_args()
    load_env()
    cfg = config(args.tag)
    eps = [f"http://127.0.0.1:{8100 + int(g)}" for g in args.gpus.split(",")]
    llm = LLM(args.tag, eps, cache_path=ROOT / "cache" / "llm" / f"{args.tag}.sqlite")
    for ds in args.datasets:
        ids = (ROOT / "results" / "e0" / "ids" / f"{ds}_{args.split}.txt").read_text().split()
        pipe = Pipeline(llm, BM25Index(ds), seed=OP_SEED, k=cfg["k"], txt=cfg["txt"], sink_q=cfg.get("sink_q", False))
        out = ROOT / "results" / "e3" / args.tag / args.split / ds / "questions.jsonl"
        materialize(ds, SPLITS[args.split], ids, pipe, out, args.workers, grid=args.split == "cal")


if __name__ == "__main__":
    main()
