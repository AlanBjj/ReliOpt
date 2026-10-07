"""e9: which questions the method escalates, at the main operating point (Section "Which questions are escalated").

Per backbone and dataset, by question structure (HotpotQA bridge / comparison, the four 2WikiMultiHopQA types, MuSiQue
2 / 3 / 4 hops): the share of questions escalated to each alternative, and, over the escalated questions, the EM the
alternative gains and loses against the method without escalation (mean over seeds, so a question counts fractionally).
Operating point: plan signal, E9_MODE, budget B_IRCoT/2, maps E9_SRC (scripts/make_table1.py).
Input:  results/e9/<tag>/<E9_SRC>.json (per-question choices), results/e3/<tag>/test/<ds>/{questions.jsonl,runs/full_s*},
        results/e3/<tag>/prices.json, results/e2/<tag>/test/<ds>/{ircot_k*,searcho1_s*}.jsonl
Output: results/e9/<tag>/escalated.{json,md}
Usage:  python3 scripts/e9_escalated.py [--tags llama31-8b qwen3-8b] [--budget B_IRCoT/2]      standard library only
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_table1 import E9_MODE, E9_SRC  # noqa: E402

DATASETS = ["hotpotqa", "2wikimultihopqa", "musique"]
SEEDS = [17, 29, 43]


def rd(p):
    return [json.loads(line) for line in open(p)]


def group(ds, qtype):
    if ds == "musique":
        return f"{qtype[0]} hops"
    return qtype


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags", nargs="+", default=["llama31-8b", "qwen3-8b"])
    ap.add_argument("--budget", default="B_IRCoT/2")
    args = ap.parse_args()
    for tag in args.tags:
        e3, e2 = ROOT / "results" / "e3" / tag, ROOT / "results" / "e2" / tag / "test"
        priced = json.loads((ROOT / "results" / "e9" / tag / f"{E9_SRC}.json").read_text())
        prices = json.loads((e3 / "prices.json").read_text())
        out, md = {}, [f"# Escalated questions ({tag}, plan/{E9_MODE}/{args.budget}, maps {E9_SRC})", "",
                       "| dataset | structure | n | to IRCoT | to Search-o1 | EM gained | EM lost | net (EM points of the group) |",
                       "|---|---|---|---|---|---|---|---|"]
        for ds in DATASETS:
            if ds not in priced:
                continue
            choice = priced[ds]["points"][f"plan/{E9_MODE}/{args.budget}"]["choice"]
            qtype = {q["qid"]: group(ds, q["qtype"]) for q in rd(e3 / "test" / ds / "questions.jsonl")}
            lam = prices[ds]["B_SC"]
            ours = {}
            for s in SEEDS:
                for r in rd(e3 / "test" / ds / "runs" / f"full_s{s}.jsonl"):
                    if abs(r["lam"] - lam) < 1e-12:
                        ours[r["qid"]] = ours.get(r["qid"], 0) + r["em"] / len(SEEDS)
            irc = {r["qid"]: r["em"] for r in rd(next(e2.glob(f"{ds}/ircot_k*.jsonl")))}
            so1 = {}
            for s in SEEDS:
                for r in rd(e2 / ds / f"searcho1_s{s}.jsonl"):
                    so1[r["qid"]] = so1.get(r["qid"], 0) + r["em"] / len(SEEDS)
            alt = {"ircot": irc, "searcho1": so1}
            rows = {}
            for q, g in qtype.items():
                if q not in ours:
                    continue
                r = rows.setdefault(g, {"n": 0, "ircot": 0, "searcho1": 0, "gain": 0.0, "loss": 0.0})
                r["n"] += 1
                a = choice.get(q)
                if a:
                    r[a] += 1
                    d = alt[a][q] - ours[q]
                    r["gain"] += max(d, 0.0)
                    r["loss"] += max(-d, 0.0)
            out[ds] = rows
            for g, r in sorted(rows.items()):
                md.append(f"| {ds} | {g} | {r['n']} | {100 * r['ircot'] / r['n']:.0f}% | {100 * r['searcho1'] / r['n']:.0f}% | "
                          f"{r['gain']:.1f} | {r['loss']:.1f} | {100 * (r['gain'] - r['loss']) / r['n']:+.1f} |")
        (ROOT / "results" / "e9" / tag / "escalated.json").write_text(json.dumps(out, indent=1))
        (ROOT / "results" / "e9" / tag / "escalated.md").write_text("\n".join(md) + "\n")
        print("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
