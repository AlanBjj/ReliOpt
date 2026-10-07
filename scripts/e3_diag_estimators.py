"""e3 diagnostics by estimator: AUROC and ECE of the step error estimate on the labelled test steps, for the type prior
(eps-bar of the step's type on the calibration set), the method's logistic regression and the gradient-boosted trees.

Why: every results table is tinted against a stated reference row, and the diagnostics table
uses the type prior as its reference, so it needs the prior's and the boosted trees' numbers on the same steps as the
logistic regression of scripts/e3_summarize.py (whose diagnostics this reproduces for "logreg").
Input:  results/e3/<tag>/test/<ds>/questions.jsonl, results/e3/<tag>/models/<ds>_{logreg,gbdt}.pkl
Output: results/e3/<tag>/test/diag_estimators.{json,md} (summary.json is left untouched)
Usage:  python scripts/e3_diag_estimators.py --tag llama31-8b      (needs scikit-learn, the project venv)
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from e3_summarize import DATASETS, ece, read  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    base = ROOT / "results" / "e3" / args.tag
    out = {}
    for ds in DATASETS:
        q = base / "test" / ds / "questions.jsonl"
        if not q.exists():
            continue
        st = [s for r in read(q) for s in r["steps"] if s["label"] in ("correct", "error")]
        y = [int(s["label"] == "error") for s in st]
        lr = pickle.load(open(base / "models" / f"{ds}_logreg.pkl", "rb"))
        gb = pickle.load(open(base / "models" / f"{ds}_gbdt.pkl", "rb"))
        preds = {"prior": [lr.prior(s["kind"]) for s in st],
                 "logreg": lr.eps([s["feat"] for s in st]),
                 "gbdt": gb.eps([s["feat"] for s in st])}
        out[ds] = {k: {"auroc": roc_auc_score(y, p), "ece": ece(p, y)} for k, p in preds.items()}
        out[ds]["n_steps"] = len(st)
        out[ds]["error_rate"] = sum(y) / len(y)
    (base / "test" / "diag_estimators.json").write_text(json.dumps(out, indent=1))
    lines = [f"# e3 diagnostics by estimator ({args.tag}, test)", "",
             "| Dataset | steps | error rate | prior AUROC / ECE | logreg AUROC / ECE | gbdt AUROC / ECE |",
             "|---|---|---|---|---|---|"]
    for ds, d in out.items():
        lines.append(f"| {ds} | {d['n_steps']} | {d['error_rate']:.3f} | " + " | ".join(
            f"{d[k]['auroc']:.3f} / {d[k]['ece']:.3f}" for k in ("prior", "logreg", "gbdt")) + " |")
    (base / "test" / "diag_estimators.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
