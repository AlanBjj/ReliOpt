"""e9: Adaptive-RAG at a matched budget for the main table.

The T5-large router's class probabilities are kept (results/e9/<tag>/arag/pred_abc.jsonl). Matched version: a question
goes to IRCoT (C) only if P(C) >= t, otherwise to the more probable of no retrieval (A) and single-step RAG (B); t is
swept on a grid of 0.001 and, per dataset, the t whose mean tokens per question are closest to the method's (the main-table
row: plan signal, E9_MODE, B_IRCoT/2, E9_SRC) is taken, as for IRCoT's matched budget. Only costs, no test label, enter
the choice of t. t = 0 sends everything to IRCoT; the official recipe is the argmax over A / B / C.
Input:  results/e9/<tag>/arag/pred_abc.jsonl, results/e2/<tag>/test/<ds>/{simple_s17,ircot_k*}.jsonl,
        results/e9/<tag>/<E9_SRC>.json
Output: results/e9/<tag>/arag/matched.json {ds: {t, em, f1, tok, cost_x, share_ircot, target_tok}} and a printed table
Usage:  python3 scripts/e9_arag_matched.py [--tags llama31-8b qwen3-8b]      standard library only
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_table1 import E9_MODE, E9_SRC  # noqa: E402

DATASETS = ["hotpotqa", "2wikimultihopqa", "musique"]


def rd(p):
    return [json.loads(line) for line in open(p)]


def tok(r):
    return sum(r["tok"].values()) if isinstance(r["tok"], dict) else r["tok"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags", nargs="+", default=["llama31-8b", "qwen3-8b"])
    args = ap.parse_args()
    for tag in args.tags:
        pred = {r["qid"]: r["probs"] for r in rd(ROOT / "results" / "e9" / tag / "arag" / "pred_abc.jsonl")}
        priced = json.loads((ROOT / "results" / "e9" / tag / f"{E9_SRC}.json").read_text())
        out = {}
        for ds in DATASETS:
            d = ROOT / "results" / "e2" / tag / "test" / ds
            simple = {r["qid"]: r for r in rd(d / "simple_s17.jsonl")}
            irc = {r["qid"]: r for r in rd(next(d.glob("ircot_k*.jsonl")))}
            qs = [q for q in irc if q in simple and q in pred]
            rag_tok = sum(tok(simple[q]["rag"]) for q in qs) / len(qs)
            target = priced[ds]["points"][f"plan/{E9_MODE}/B_IRCoT/2"]["tok"]

            def run(t):
                em = f1 = tk = n_c = 0.0
                for q in qs:
                    p = pred[q]
                    if p["C"] >= t:
                        r, n_c = irc[q], n_c + 1
                    else:
                        r = simple[q]["cot"] if p["A"] >= p["B"] else simple[q]["rag"]
                    em += r["em"]; f1 += r["f1"]; tk += tok(r)
                n = len(qs)
                return {"t": t, "em": 100 * em / n, "f1": 100 * f1 / n, "tok": tk / n, "cost_x": tk / n / rag_tok,
                        "share_ircot": n_c / n}

            best = min((run(i / 1000) for i in range(0, 1001)), key=lambda r: abs(r["tok"] - target))
            out[ds] = {**best, "target_tok": target}
            print(f"{tag:11s} {ds:16s} target {target / 1000:5.1f}k  t={best['t']:.3f}  EM {best['em']:5.1f}  F1 {best['f1']:5.1f}"
                  f"  tok {best['tok'] / 1000:5.1f}k  to IRCoT {100 * best['share_ircot']:3.0f}%")
        (ROOT / "results" / "e9" / tag / "arag" / "matched.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
