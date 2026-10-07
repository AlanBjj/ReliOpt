"""e9: how complementary the typed plan and IRCoT are on the test subsets (introduction): per backbone and dataset, EM of
Plan-Only, of IRCoT's official setting, and of an oracle that takes whichever of the two is right, and the oracle's margin
over the better of the two.
Input:  results/e3/<tag>/test/<ds>/runs/plan_only_s17.jsonl, results/e2/<tag>/test/<ds>/ircot_k*.jsonl
Output: results/e9/complementarity.json and a printed table
Usage:  python3 scripts/e9_complementarity.py      standard library only
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rd(p):
    return {r["qid"]: r["em"] for r in map(json.loads, open(p))}


def main():
    out = {}
    for tag in ("llama31-8b", "qwen3-8b"):
        for ds in ("hotpotqa", "2wikimultihopqa", "musique"):
            po = rd(ROOT / "results" / "e3" / tag / "test" / ds / "runs" / "plan_only_s17.jsonl")
            irc = rd(next((ROOT / "results" / "e2" / tag / "test" / ds).glob("ircot_k*.jsonl")))
            qs = sorted(set(po) & set(irc))
            plan, ircot = (100 * sum(d[q] for q in qs) / len(qs) for d in (po, irc))
            either = 100 * sum(max(po[q], irc[q]) for q in qs) / len(qs)
            out[f"{tag}/{ds}"] = {"n": len(qs), "plan": plan, "ircot": ircot, "either": either,
                                  "margin": either - max(plan, ircot)}
            print(f"{tag:11s} {ds:16s} plan {plan:5.1f} ircot {ircot:5.1f} either {either:5.1f} margin {either - max(plan, ircot):5.1f}")
    m = [v["margin"] for v in out.values()]
    print(f"margin range {min(m):.1f} - {max(m):.1f}")
    (ROOT / "results" / "e9" / "complementarity.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
