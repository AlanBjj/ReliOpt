"""e0: check that the local vLLM servers behave as the pipeline assumes, and measure throughput.

Input:  running servers from run/e0_serve.sh (ports 8100 + GPU id)
Output: results/e0/smoke_<tag>.json
Usage:  python scripts/e0_smoke_llm.py --tag llama31-8b --gpus 0,2,3 [--n 300 --workers 96]
Not resumable on purpose: it is a short measurement; rerunning overwrites the json.

Checks: greedy decoding is repeatable; token logprobs come back; n samples with a seed come back; the Qwen3 non-thinking
switch suppresses <think> blocks. Throughput uses synthetic ~700-token prompts shaped like a retrieval step (5 passages +
a sub-question, short answer), without the cache, all GPUs together.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reliopt.llm.client import LLM, run_parallel  # noqa: E402
from reliopt.utils import load_env  # noqa: E402

WORDS = ("film director born city river company founded year album band member award novel author capital country "
         "university team season league museum painter village province actor singer series episode").split()


def passage(rng, n_words=120):
    return " ".join(rng.choice(WORDS) for _ in range(n_words)).capitalize() + "."


def make_prompt(rng, i):
    ctx = "\n\n".join(f"[{j + 1}] Title {rng.randint(1, 10 ** 6)}: {passage(rng)}" for j in range(5))
    q = f"Question {i}: in which year was the {rng.choice(WORDS)} of the {rng.choice(WORDS)} founded?"
    return [{"role": "system", "content": "Answer using only the passages. Reply with a short answer."},
            {"role": "user", "content": f"{ctx}\n\n{q}\nAnswer:"}]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpus", required=True, help="comma-separated GPU ids used by run/e0_serve.sh")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=96)
    args = ap.parse_args()
    load_env()
    eps = [f"http://127.0.0.1:{8100 + int(g)}" for g in args.gpus.split(",")]
    llm = LLM(args.tag, eps, cache_path=None)
    rng = random.Random(0)
    msgs = [{"role": "user", "content": "Who wrote the novel 'Pride and Prejudice'? Answer in a few words."}]
    rep = {"tag": args.tag, "endpoints": eps, "checks": {}}

    a, b = llm.chat(msgs, max_tokens=32), llm.chat(msgs, max_tokens=32)
    rep["checks"]["greedy_repeatable"] = a["texts"] == b["texts"]
    rep["checks"]["greedy_answer"] = a["texts"][0][:200]
    lp = llm.chat(msgs, max_tokens=32, logprobs=True)
    rep["checks"]["logprobs_len_matches_completion"] = (lp["logprobs"] is not None
                                                        and len(lp["logprobs"][0]) == lp["usage"]["completion"])
    s1 = llm.chat(msgs, temperature=0.7, top_p=0.95, max_tokens=32, n=5, seed=17)
    s2 = llm.chat(msgs, temperature=0.7, top_p=0.95, max_tokens=32, n=5, seed=17)
    rep["checks"]["n5_returned"] = len(s1["texts"]) == 5
    rep["checks"]["seeded_sampling_repeatable"] = s1["texts"] == s2["texts"]
    rep["checks"]["n5_usage"] = s1["usage"]
    if args.tag.startswith("qwen3"):
        rep["checks"]["no_think_block"] = all("<think>" not in t for t in a["texts"] + s1["texts"])

    prompts = [make_prompt(rng, i) for i in range(args.n)]
    llm.stats = {k: 0 for k in llm.stats}
    t0 = time.time()
    outs = run_parallel(lambda m: llm.chat(m, max_tokens=48), prompts, workers=args.workers)
    secs = time.time() - t0
    st = llm.stats
    rep["throughput"] = {"requests": len(outs), "seconds": round(secs, 1), "prompt_tokens": st["prompt"],
                         "completion_tokens": st["completion"],
                         "prompt_tok_per_s": round(st["prompt"] / secs), "completion_tok_per_s": round(st["completion"] / secs),
                         "requests_per_s": round(len(outs) / secs, 2), "n_gpus": len(eps),
                         "mean_prompt_tokens": round(st["prompt"] / len(outs)),
                         "example_output": outs[0]["texts"][0][:120]}
    out = ROOT / "results" / "e0" / f"smoke_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    bad = [k for k, v in rep["checks"].items() if v is False]
    if bad:
        sys.exit(f"FAILED checks: {bad}")


if __name__ == "__main__":
    main()
