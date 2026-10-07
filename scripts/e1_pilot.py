"""e1 pilot: does step-level verification have headroom, and is its value uneven across steps?

Input:  results/e0/ids/<ds>_cal.txt (calibration questions from IRCoT's processed train split), cache/bm25/<ds>/,
        vLLM servers from run/e0_serve.sh (ports 8100 + GPU id)
Output: results/e1/ids/<ds>.txt          the pilot questions: a fixed subsample of the calibration ids (so e3 reuses the cache)
        results/e1/<ds>/questions.jsonl  one row per question with
          plan     the parsed plan (fallback flag, attempts, tokens)
          base     the plan-only run (prediction, EM/F1, tokens)
          steps    per step: resolved sub-question, default answer, label, free features, and every operator applied
                   standalone on the plan-only inputs (answer, tokens, label of the operator's answer)
          single   end-to-end runs that apply one answer-changing operator at one step (only where it changes the answer)
          grid     end-to-end runs of every type-level policy: (TXT, CMP, LOG) each in NONE/VOTE3/VOTE5/ALT/VOTE5+ALT
        cache/llm/<tag>.sqlite (server only): every model call, shared with later experiments
Usage:  python scripts/e1_pilot.py --tag llama31-8b --gpus 0,1,2,3 [--datasets 2wikimultihopqa musique --n 200] [--run r2]
            [--txt short|span --k 5]   (TXT reader chosen on the tuning split by scripts/e1_txt_tune.py from round 4)
        --run names a later round after prompt or operator changes: its rows go to results/e1/<run>/<ds>/ (round 1 stays in
        results/e1/<ds>/); the pilot ids are shared by all rounds.
Resumable: rows go to questions.partial.jsonl as questions finish and finished questions are skipped; the file is renamed to
questions.jsonl only when every question is there. Model calls are cached, so a rerun repeats no finished call.
"""
import argparse
import itertools
import json
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reliopt.data.ircot import load_questions  # noqa: E402
from reliopt.exec.operators import GENERIC, OPS, answer_changing  # noqa: E402
from reliopt.exec.replay import NO_VERIFY, Pipeline, at_step, static  # noqa: E402
from reliopt.exec.steps import tokens_of  # noqa: E402
from reliopt.llm.client import LLM  # noqa: E402
from reliopt.plan.planner import structure  # noqa: E402
from reliopt.quality.features import step_features  # noqa: E402
from reliopt.quality.labels import gold_view, label_answer, label_run  # noqa: E402
from reliopt.retrieval.bm25 import BM25Index  # noqa: E402
from reliopt.utils import load_env  # noqa: E402

E1_SEED = 1007   # subsample of the calibration ids
OP_SEED = 17     # pool sampling seed (first of the three seeds 17, 29, 43)


def pilot_ids(ds, n):
    path = ROOT / "results" / "e1" / "ids" / f"{ds}.txt"
    if path.exists():
        return path.read_text().split()
    cal = (ROOT / "results" / "e0" / "ids" / f"{ds}_cal.txt").read_text().split()
    ids = sorted(random.Random(E1_SEED).sample(cal, n))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n")
    return ids


def compact(r):
    return {"pred": r["pred"], "em": r["em"], "f1": r["f1"], "tok": r["tok"], "calls": r["calls"]}


def process(pipe, q, with_grid=True):
    """Materialise one question; with_grid=False skips the type-level policy grid (not needed on test questions)."""
    t0 = time.time()
    plan = pipe.plan(q["question"])
    base = pipe.run(q, plan, NO_VERIFY, keep=True)
    struct = structure(plan["steps"])
    view = gold_view(q)
    labels = label_run(view, plan, base)
    sink = plan["steps"][-1]["id"]
    steps = []
    for s in plan["steps"]:
        sid, kind = s["id"], s["kind"]
        ctx, r0 = base["_ctx"][sid], base["_r0"][sid]
        parents_ok = all(labels[d] == "correct" for d in s["deps"])
        ops = {}
        for op in OPS[kind][1:]:
            res = pipe.ops.apply(op, ctx, r0)
            info = {k: v for k, v in res["info"].items() if k in ("agree", "probe", "new", "verdict", "vote", "query2")}
            ops[op] = {"ans": res["answer"], "tok": tokens_of(res["cost"]), "calls": res["cost"]["calls"],
                       "label": label_answer(view, kind, ctx["resolved"], res["answer"], parents_ok, sid == sink), **info}
        row = {"id": sid, "kind": kind, "deps": s["deps"], "q": ctx["resolved"], "a0": r0["answer"],
               "label": labels[sid], "exec_tok": tokens_of(r0["cost"]), "feat": step_features(struct[sid], ctx, r0),
               "ops": ops}
        if kind == "TXT":
            row["titles"] = [h["title"] for h in r0["hits"]]
            row["cited"] = r0.get("cited", [])
        if kind == "CMP":
            row["exec_ok"], row["fallback"] = r0.get("exec_ok"), r0.get("fallback")
        steps.append(row)
    single = {}
    for row in steps:
        for op in answer_changing(row["kind"]):
            if row["ops"][op]["ans"] != row["a0"]:
                single[f"{row['id']}:{op}"] = compact(pipe.run(q, plan, at_step(row["id"], op)))
    grid, seen = {}, {}
    for t, c, l in itertools.product(GENERIC, repeat=3) if with_grid else []:
        pol = static({"TXT": t, "CMP": c, "LOG": l})
        key = tuple(pol(s) for s in plan["steps"])   # policies that coincide on this plan are run once
        if key not in seen:
            seen[key] = compact(pipe.run(q, plan, pol))
        grid[f"{t}|{c}|{l}"] = seen[key]
    return {"qid": q["qid"], "question": q["question"], "answers": q["answers"], "qtype": q["qtype"],
            "plan": {"steps": plan["steps"], "fallback": plan["fallback"], "attempts": plan["attempts"],
                     "error": plan["error"], "tok": tokens_of(plan["cost"]),
                     "raw": plan["raw"] if plan["fallback"] or plan["attempts"] > 1 else None},
            "base": compact(base), "steps": steps, "single": single, "grid": grid,
            "seconds": round(time.time() - t0, 1)}


def run_dataset(ds, args, llm):
    out = ROOT / "results" / "e1" / args.run / ds / "questions.jsonl"
    ids = pilot_ids(ds, args.n)
    pipe = Pipeline(llm, BM25Index(ds), seed=OP_SEED, k=args.k, txt=args.txt)
    materialize(ds, "train", ids, pipe, out, args.workers)


def materialize(ds, split, ids, pipe, out, workers, grid=True):
    """process() every question of `ids` into `out` (resumable through questions.partial.jsonl)."""
    if out.exists():
        print(f"skip {ds}: {out} exists"); return
    part = out.with_name(out.stem + ".partial.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["qid"] for l in part.open()} if part.exists() else set()
    t0 = time.time()
    qs = load_questions(ds, split, qids=ids)
    print(f"{ds}: {len(qs)} questions loaded in {time.time() - t0:.0f}s, {len(done)} already done"); sys.stdout.flush()
    assert len(qs) == len(ids), f"{ds}: found {len(qs)} of {len(ids)} ids in the {split} split"
    llm = pipe.llm
    todo = [q for q in qs if q["qid"] not in done]
    lock, n_ok, n_err = threading.Lock(), 0, 0
    with part.open("a") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(process, pipe, q, with_grid=grid): q["qid"] for q in todo}
        for fut in as_completed(futs):
            try:
                row = fut.result()
            except Exception:
                n_err += 1
                print(f"ERROR {ds} {futs[fut]}:\n{traceback.format_exc()}"); sys.stdout.flush()
                continue
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            n_ok += 1
            if n_ok % 20 == 0 or n_ok == len(todo):
                st = llm.stats
                print(f"{ds}: {len(done) + n_ok}/{len(qs)} done, {time.time() - t0:.0f}s, calls {st['calls']} "
                      f"(cache hits {st['cache_hits']})"); sys.stdout.flush()
    n = sum(1 for _ in part.open())
    if n != len(qs):
        sys.exit(f"{ds}: only {n}/{len(qs)} rows in {part} ({n_err} errors above); rerun the same command to continue")
    part.rename(out)
    print(f"wrote {out} ({n} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpus", required=True, help="comma-separated GPU ids whose vLLM servers to use")
    ap.add_argument("--datasets", nargs="+", default=["2wikimultihopqa", "musique"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=160)
    ap.add_argument("--run", default="", help="round name; empty = round 1 in results/e1/<ds>/")
    ap.add_argument("--txt", default="short", choices=["short", "span"], help="TXT reader style (span from round 4)")
    ap.add_argument("--k", type=int, default=5, help="TXT retrieval depth")
    args = ap.parse_args()
    load_env()
    eps = [f"http://127.0.0.1:{8100 + int(g)}" for g in args.gpus.split(",")]
    cache = ROOT / "cache" / "llm" / f"{args.tag}.sqlite"
    cache.parent.mkdir(parents=True, exist_ok=True)
    llm = LLM(args.tag, eps, cache_path=cache)
    for ds in args.datasets:
        run_dataset(ds, args, llm)


if __name__ == "__main__":
    main()
