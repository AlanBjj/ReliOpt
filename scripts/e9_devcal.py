"""e9: an in-distribution calibration set for escalation (500 development questions per dataset, disjoint from test and tune).

Why (2026-10-07 23:30): the escalation maps P(plan right | s) and P(alternative right | s) fitted on the 500 calibration
questions of e3 (training split) did not transfer: on the training questions IRCoT beats the plan on HotpotQA (Qwen3-8B:
56.6 vs 54.4 EM), on the test subsets it does not (44.4 vs 45.2), so priced escalation over-escalated at large budgets.
The IRCoT test subsets are sampled from the development sets, so the escalation maps are fitted on further development
questions instead. The quality model (eps-hat, pi-hat) stays the one fitted on the training calibration set; here the plan
is only planned and executed by default (no operator), which gives the reliability signal and the plan's answer.
  --make-ids  sample 500 ids per dataset from <ds>/dev.jsonl minus the test and tune ids (seed 2026), write
              results/e0/ids/<ds>_devcal.txt (refuses to overwrite); with --split devcal2, 500 more minus devcal too
              (seed 2027, results/e0/ids/<ds>_devcal2.txt;: enlarge the calibration set)
  (default)   plan + default execution of every devcal question: results/e9/<tag>/devcal/<ds>/questions.jsonl, rows as in
              e3's questions.jsonl but only qid, question, answers, plan, base and steps (id, kind, deps, a0, feat)
Input:  cache/ircot/processed_data/<ds>/dev.jsonl, results/e0/ids/<ds>_{test,tune}.txt, results/e3/<tag>/config.json,
        cache/bm25/<ds>/, a running vLLM server for <tag>
Usage:  python scripts/e9_devcal.py --make-ids [--split devcal2]
        python scripts/e9_devcal.py --tag qwen3-8b --gpus 1,2 [--split devcal2]   (resumable: finished questions are kept)
"""
import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DATASETS = ["hotpotqa", "2wikimultihopqa", "musique"]
N = 500
# devcal2: 500 more, disjoint from devcal too
SPLITS = {"devcal": {"seed": 2026, "exclude": ("test", "tune")}, "devcal2": {"seed": 2027, "exclude": ("test", "tune", "devcal")}}


def make_ids(split):
    for ds in DATASETS:
        out = ROOT / "results" / "e0" / "ids" / f"{ds}_{split}.txt"
        if out.exists():
            print(f"{out} exists, kept"); continue
        taken = set()
        for sp in SPLITS[split]["exclude"]:
            taken |= set((ROOT / "results" / "e0" / "ids" / f"{ds}_{sp}.txt").read_text().split())
        pool = []
        with open(ROOT / "cache" / "ircot" / "processed_data" / ds / "dev.jsonl") as f:
            for line in f:
                qid = json.loads(line)["question_id"]
                if qid not in taken:
                    pool.append(qid)
        ids = sorted(random.Random(SPLITS[split]["seed"]).sample(sorted(pool), N))
        out.write_text("\n".join(ids) + "\n")
        print(f"{ds}: {len(pool)} candidate dev questions, {len(ids)} sampled -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--make-ids", action="store_true")
    ap.add_argument("--split", choices=list(SPLITS), default="devcal")
    ap.add_argument("--tag")
    ap.add_argument("--gpus")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--workers", type=int, default=96)
    args = ap.parse_args()
    if args.make_ids:
        make_ids(args.split); return
    from reliopt.data.ircot import load_questions
    from reliopt.exec.replay import NO_VERIFY, Pipeline
    from reliopt.llm.client import LLM
    from reliopt.retrieval.bm25 import BM25Index
    from reliopt.utils import load_env
    from e1_pilot import OP_SEED, step_features, structure, tokens_of
    load_env()
    cfg = json.loads((ROOT / "results" / "e3" / args.tag / "config.json").read_text())
    eps = [f"http://127.0.0.1:{8100 + int(g)}" for g in args.gpus.split(",")]
    llm = LLM(args.tag, eps, cache_path=ROOT / "cache" / "llm" / f"{args.tag}.sqlite")
    for ds in args.datasets:
        out = ROOT / "results" / "e9" / args.tag / args.split / ds / "questions.jsonl"
        if out.exists():
            print(f"{ds}: {out} exists, skipped"); continue
        part = out.with_name("questions.partial.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        done = {json.loads(l)["qid"] for l in part.open()} if part.exists() else set()
        ids = (ROOT / "results" / "e0" / "ids" / f"{ds}_{args.split}.txt").read_text().split()
        qs = load_questions(ds, "dev", qids=ids)
        assert len(qs) == len(ids), (ds, len(qs), len(ids))
        pipe = Pipeline(llm, BM25Index(ds), seed=OP_SEED, k=cfg["k"], txt=cfg["txt"], sink_q=cfg.get("sink_q", False))
        lock, t0, n = threading.Lock(), time.time(), 0

        def one(q):
            plan = pipe.plan(q["question"])
            base = pipe.run(q, plan, NO_VERIFY, keep=True)
            struct = structure(plan["steps"])
            steps = [{"id": s["id"], "kind": s["kind"], "deps": s["deps"], "a0": base["_r0"][s["id"]]["answer"],
                      "feat": step_features(struct[s["id"]], base["_ctx"][s["id"]], base["_r0"][s["id"]])}
                     for s in plan["steps"]]
            return {"qid": q["qid"], "question": q["question"], "answers": q["answers"],
                    "plan": {"steps": plan["steps"], "fallback": plan["fallback"], "tok": tokens_of(plan["cost"])},
                    "base": {"pred": base["pred"], "em": base["em"], "f1": base["f1"], "tok": base["tok"]}, "steps": steps}

        todo = [q for q in qs if q["qid"] not in done]
        with part.open("a") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
            for fut in as_completed([pool.submit(one, q) for q in todo]):
                try:
                    row = fut.result()
                except Exception as e:   # left out of the partial file, so a rerun redoes it
                    print(f"{ds}: error {type(e).__name__}: {str(e)[:200]}"); continue
                with lock:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush(); n += 1
                    if n % 100 == 0:
                        print(f"{ds}: {len(done) + n}/{len(qs)} done, {time.time() - t0:.0f}s"); sys.stdout.flush()
        rows = {json.loads(l)["qid"]: l for l in part.open()}
        if len(rows) == len(qs):
            out.write_text("".join(rows[q["qid"]] for q in qs))
            print(f"{ds}: wrote {out} ({len(qs)} rows), EM {100 * sum(json.loads(r)['base']['em'] for r in rows.values()) / len(qs):.1f}")
        else:
            print(f"{ds}: {len(rows)}/{len(qs)} finished; rerun to complete")


if __name__ == "__main__":
    main()
