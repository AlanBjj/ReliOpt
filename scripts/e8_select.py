"""Route-2 quick check: choose the execution configuration per dataset on the calibration questions, score it on test.

Why: e8 showed that the execution-layer choices help on some datasets and hurt on others, so the
paper may let the optimizer pick physical implementations too (route 2). Before committing, check that a choice made on
calibration data transfers: per backbone and dataset, take the e8 variant with the highest EM on the 500 calibration
questions (ties keep "full"), and read that variant's test EM and its paired bootstrap against "full" from the e8 test
summary. Go criterion: on both backbones, the chosen configuration is at least 1.5 EM above "full" on
the test set for at least two of the three datasets.
Input:  results/e8/<tag>/cal/summary.json and results/e8/<tag>/summary.json (scripts/e8_summarize.py with and without --split cal)
Output: results/e8/select.{json,md}
Usage:  python3 scripts/e8_select.py [--tags llama31-8b qwen3-8b]      standard library only
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["hotpotqa", "2wikimultihopqa", "musique"]


def load(p):
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags", nargs="+", default=["llama31-8b", "qwen3-8b"])
    args = ap.parse_args()
    out, md = {}, ["# Route-2 quick check: execution configuration chosen on calibration, scored on test", "",
                   "| Backbone | Dataset | chosen on cal | cal EM full → chosen | test EM full → chosen | test ΔEM | 95% CI |",
                   "|---|---|---|---|---|---|---|"]
    for tag in args.tags:
        cal = load(ROOT / "results" / "e8" / tag / "cal" / "summary.json")
        test = load(ROOT / "results" / "e8" / tag / "summary.json")
        out[tag] = {}
        for ds in DATASETS:
            c, t = cal.get(ds, {}), test.get(ds, {})
            if "full" not in c or "full" not in t:
                md.append(f"| {tag} | {ds} | (missing) | | | | |")
                continue
            best = max(c, key=lambda v: (round(c[v]["em"], 6), v == "full"))
            tv = t.get(best, {})
            d = tv.get("d_em", 0.0) if best != "full" else 0.0
            ci = f"[{tv['lo']:+.1f}, {tv['hi']:+.1f}]" if best != "full" and "lo" in tv else "—"
            out[tag][ds] = {"chosen": best, "cal_full": c["full"]["em"], "cal_chosen": c[best]["em"],
                            "test_full": t["full"]["em"], "test_chosen": tv.get("em"), "test_d_em": d,
                            "test_lo": tv.get("lo"), "test_hi": tv.get("hi")}
            md.append(f"| {tag} | {ds} | {best} | {c['full']['em']:.1f} → {c[best]['em']:.1f} | "
                      f"{t['full']['em']:.1f} → {tv.get('em', float('nan')):.1f} | {d:+.1f} | {ci} |")
    go = {tag: sum(1 for v in out[tag].values() if v["test_d_em"] >= 1.5) for tag in out}
    md += ["", "Datasets with test ΔEM ≥ 1.5 per backbone: " + ", ".join(f"{k} {v}/3" for k, v in go.items()),
           "Go (route 2) if every backbone has at least 2: " + str(bool(go) and all(v >= 2 for v in go.values()))]
    (ROOT / "results" / "e8").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "e8" / "select.json").write_text(json.dumps({"by_tag": out, "count": go}, indent=1))
    (ROOT / "results" / "e8" / "select.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
