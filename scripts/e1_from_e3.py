"""Round 4 of the e1 pilot without new model calls: the pilot questions are a subset of the calibration questions, so their
rows are taken from the e3 calibration materialisation (same process(), round-4 configuration in results/e3/<tag>/config.json).

Input:  results/e1/ids/<ds>.txt, results/e3/<tag>/cal/<ds>/questions.jsonl
Output: results/e1/<run>/<ds>/questions.jsonl, then run scripts/e1_analyze.py --run <run> and scripts/e1_allocsim.py --run <run>
Usage:  python scripts/e1_from_e3.py --tag llama31-8b --run r4
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--run", default="r4")
    ap.add_argument("--datasets", nargs="+", default=["2wikimultihopqa", "musique"])
    args = ap.parse_args()
    for ds in args.datasets:
        ids = (ROOT / "results" / "e1" / "ids" / f"{ds}.txt").read_text().split()
        rows = {json.loads(l)["qid"]: l for l in open(ROOT / "results" / "e3" / args.tag / "cal" / ds / "questions.jsonl")}
        missing = [i for i in ids if i not in rows]
        assert not missing, f"{ds}: {len(missing)} pilot ids not in the calibration rows"
        out = ROOT / "results" / "e1" / args.run / ds / "questions.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(rows[i] if rows[i].endswith("\n") else rows[i] + "\n" for i in ids))
        print(f"{ds}: {len(ids)} pilot rows -> {out}")
    cfg = json.loads((ROOT / "results" / "e3" / args.tag / "config.json").read_text())
    (ROOT / "results" / "e1" / args.run / "config.json").write_text(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
