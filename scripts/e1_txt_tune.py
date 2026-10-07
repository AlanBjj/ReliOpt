"""Round-4 choice of the TXT reader on the tuning split: answer style ("short" up to round 3, "span" = copied word for word)
and retrieval depth k, at most three configurations as for every method.

Each configuration runs the plan-only pipeline (no verification) on IRCoT's 100 tuning questions per dataset; the plan is
generated once and shared. The IRCoT tuning runs on the same questions (results/e2/<tag>/tune/) are printed alongside.
Input:  results/e0/ids/<ds>_tune.txt, cache/bm25/<ds>/, vLLM servers (ports 8100 + GPU id)
Output: results/e1/r4/<name>/<ds>_<style>_k<k>.jsonl (one row per question), results/e1/r4/<name>/summary.md
        (<name> = "tune" for the round-3 planner, "tune_plan2" after the round-4 planner fix)
Usage:  python scripts/e1_txt_tune.py --tag llama31-8b --gpus 0,1,2,3 [--configs short:5 span:5 span:8 span:8:q]
            [--split tune|cal --n 100 --name tune]     (":q" = the TXT sink also sees and retrieves for the question)
Resumable through the LLM cache; a finished output file is skipped.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reliopt.data.ircot import load_questions  # noqa: E402
from reliopt.exec.replay import NO_VERIFY, Pipeline  # noqa: E402
from reliopt.llm.client import LLM, run_parallel  # noqa: E402
from reliopt.retrieval.bm25 import BM25Index  # noqa: E402
from reliopt.utils import load_env  # noqa: E402



def mean(rows, k):
    return sum(r[k] for r in rows) / len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimultihopqa", "musique"])
    ap.add_argument("--configs", nargs="+", default=["short:5", "span:5", "span:8"])
    ap.add_argument("--workers", type=int, default=100)
    ap.add_argument("--name", default="tune", help="output directory under results/e1/r4/ (a new one after a planner change)")
    ap.add_argument("--split", choices=["tune", "cal"], default="tune",
                    help="tune = IRCoT's tuning questions (selection); cal = training questions (development)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--write-config", action="store_true",
                    help="write the configuration with the best mean EM over the datasets to results/e3/<tag>/config.json")
    args = ap.parse_args()
    OUT = ROOT / "results" / "e1" / "r4" / args.name
    load_env()
    eps = [f"http://127.0.0.1:{8100 + int(g)}" for g in args.gpus.split(",")]
    llm = LLM(args.tag, eps, cache_path=ROOT / "cache" / "llm" / f"{args.tag}.sqlite")
    OUT.mkdir(parents=True, exist_ok=True)
    table = {}
    for ds in args.datasets:
        ids = (ROOT / "results" / "e0" / "ids" / f"{ds}_{args.split}.txt").read_text().split()[:args.n]
        qs = load_questions(ds, {"tune": "dev_subsampled", "cal": "train"}[args.split], qids=ids)
        index = BM25Index(ds)
        for cfg in args.configs:
            style, k, *opt = cfg.split(":")
            out = OUT / f"{ds}_{style}_k{k}{'_q' if 'q' in opt else ''}.jsonl"
            if not out.exists():
                t0 = time.time()
                pipe = Pipeline(llm, index, k=int(k), txt=style, sink_q="q" in opt)

                def one(q, pipe=pipe):
                    try:
                        plan = pipe.plan(q["question"])
                        r = pipe.run(q, plan, NO_VERIFY)
                        return {"qid": q["qid"], "pred": r["pred"], "em": r["em"], "f1": r["f1"],
                                "tok": sum(r["tok"].values()), "steps": len(plan["steps"])}
                    except Exception:
                        return {"qid": q["qid"], "pred": "", "em": 0.0, "f1": 0.0, "tok": 0, "steps": 0,
                                "error": traceback.format_exc()[-800:]}
                rows = run_parallel(one, qs, workers=args.workers)
                out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
                print(f"{ds:16s} {cfg:8s} {time.time() - t0:.0f}s, errors {sum('error' in r for r in rows)}")
            rows = [json.loads(l) for l in out.open()]
            table[(ds, cfg)] = (100 * mean(rows, "em"), 100 * mean(rows, "f1"), mean(rows, "tok"))
        for p in sorted((ROOT / "results" / "e2" / args.tag / "tune" / ds).glob("ircot_k*.jsonl")) if args.split == "tune" else []:
            rows = [json.loads(l) for l in p.open()]
            table[(ds, "IRCoT " + p.stem[len("ircot_"):])] = (100 * mean(rows, "em"), 100 * mean(rows, "f1"),
                                                              mean(rows, "tok"))
    cfgs = list(dict.fromkeys(c for _, c in table))
    lines = [f"# Round-4 TXT reader on the {args.split} split (plan-only, no verification)", "",
             f"EM / F1 in %, Tok = tokens per question; {args.n} questions per dataset ({args.split}).", "",
             "| Config | " + " | ".join(f"{ds} EM / F1 / Tok" for ds in args.datasets) + " | mean EM |",
             "|---" * (len(args.datasets) + 2) + "|"]
    for c in cfgs:
        cells = [f"{table[(ds, c)][0]:.1f} / {table[(ds, c)][1]:.1f} / {table[(ds, c)][2]:,.0f}" if (ds, c) in table else "—"
                 for ds in args.datasets]
        ems = [table[(ds, c)][0] for ds in args.datasets if (ds, c) in table]
        lines.append(f"| {c} | " + " | ".join(cells) + f" | {sum(ems) / len(ems):.1f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if args.write_config:   # the round-4 rule: best mean EM over the datasets (ties: the earlier configuration)
        means = {c: sum(table[(ds, c)][0] for ds in args.datasets) / len(args.datasets) for c in args.configs}
        best = max(args.configs, key=lambda c: (means[c], -args.configs.index(c)))
        style, k, *opt = best.split(":")
        cfg = {"txt": style, "k": int(k), "sink_q": "q" in opt,
               "chosen_on": f"{OUT.relative_to(ROOT)}/summary.md (mean EM: "
                            + ", ".join(f"{c} {means[c]:.1f}" for c in args.configs) + ")"}
        path = ROOT / "results" / "e3" / args.tag / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=1))
        print(f"wrote {path}: {cfg}")


if __name__ == "__main__":
    main()
