"""e3 step 2: fit the quality model on the materialised calibration questions and pick the
Type-Static policies (the best type-level policy of the calibration grid at each verification budget).

Input:  results/e3/<tag>/cal/<ds>/questions.jsonl (scripts/e3_materialize.py --split cal)
Output: results/e3/<tag>/models/<ds>_<estimator>.pkl (QualityModel), results/e3/<tag>/models/<ds>_summary.json
        (eps-bar, rho+/-, tau, cost table, 5-fold CV AUROC of eps-hat on the calibration steps, Type-Static frontier)
Usage:  python scripts/e3_fit.py --tag llama31-8b [--datasets ...]      CPU only, a few seconds per dataset
"""
import argparse
import json
import pickle
import random
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reliopt.exec.operators import OPS  # noqa: E402
from reliopt.quality.model import QualityModel  # noqa: E402

warnings.filterwarnings("ignore")


def cv_auroc(rows, estimator, folds=5):
    """AUROC of eps-hat on calibration steps, each question scored by a model fitted without it."""
    from sklearn.metrics import roc_auc_score
    idx = list(range(len(rows)))
    random.Random(0).shuffle(idx)
    ys, ps = [], []
    for f in range(folds):
        test = set(idx[f::folds])
        m = QualityModel(estimator=estimator).fit([rows[i] for i in idx if i not in test], OPS)
        st = [s for i in test for s in rows[i]["steps"] if s["label"] in ("correct", "error")]
        ys += [int(s["label"] == "error") for s in st]
        ps += m.eps([s["feat"] for s in st])
    return roc_auc_score(ys, ps)


def type_static_frontier(rows):
    """Calibration (verification tokens, EM) of every type-level policy, and its upper frontier."""
    pol = {}
    for k in rows[0]["grid"]:
        em = 100 * sum(r["grid"][k]["em"] for r in rows) / len(rows)
        tok = sum(r["grid"][k]["tok"]["verify"] for r in rows) / len(rows)
        pol[k] = (tok, em)
    front, best = [], -1.0
    for k, (tok, em) in sorted(pol.items(), key=lambda x: (x[1][0], -x[1][1])):
        if em > best:
            front.append((k, round(tok, 1), round(em, 2)))
            best = em
    return pol, front


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimultihopqa", "musique"])
    ap.add_argument("--estimators", nargs="+", default=["logreg", "gbdt"])
    args = ap.parse_args()
    out_dir = ROOT / "results" / "e3" / args.tag / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        rows = [json.loads(l) for l in open(ROOT / "results" / "e3" / args.tag / "cal" / ds / "questions.jsonl")]
        summ = {"n_questions": len(rows)}
        for est in args.estimators:
            m = QualityModel(estimator=est).fit(rows, OPS)
            pickle.dump(m, open(out_dir / f"{ds}_{est}.pkl", "wb"))
            summ[est] = {"cv_auroc": round(cv_auroc(rows, est), 4)}
            if est == "logreg":
                summ.update(m.summary())
        pol, front = type_static_frontier(rows)
        summ["type_static_frontier"] = front
        summ["plan_only_em"] = 100 * sum(r["base"]["em"] for r in rows) / len(rows)
        (out_dir / f"{ds}_summary.json").write_text(json.dumps(summ, indent=1))
        print(f"{ds}: {len(rows)} questions, {summ['n_steps']} decidable steps ({summ['n_err']} wrong); "
              f"CV AUROC " + ", ".join(f"{e} {summ[e]['cv_auroc']:.3f}" for e in args.estimators)
              + f"; tau {json.dumps({k: round(v, 2) for k, v in summ['tau'].items()})}")


if __name__ == "__main__":
    main()
