"""Offline check (no model calls): do evidence-agreement features, computed from a deeper BM25 retrieval at no token cost,
make the step-error estimate and the allocation better?

For every TXT step: retrieve the top-EV_K passages for its resolved sub-question (and, at a sink, also for the original
question), then
  ev_count   share of those passages that contain the step's answer (normalised whole-word match)
  ev_first   rank of the first passage containing it, divided by EV_K (1 when none does)
  ev_top3    whether one of the first 3 contains it
Retrieval runs on the CPU and calls no model, so these would be free features at run time. Other step kinds get 0 / 1 / 0.
The comparison reuses scripts/e3_offline_rho.py (5-fold CV over calibration questions, at most one verified step per
question): eps-hat AUROC, and the type_rho / uplift allocators with and without the new features.
Input:  results/e3/<tag>/cal/<ds>/questions.jsonl, cache/bm25/<ds>/, results/e2/<tag>/test/summary.json
Output: results/e3/<tag>/offline_evidence.{json,md}
Usage:  python scripts/e3_offline_evidence.py --tag llama31-8b
"""
import argparse
import json
import random
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import e3_offline_rho as off  # noqa: E402
import reliopt.quality.model as qmod  # noqa: E402
from reliopt.eval.metrics import normalize_answer  # noqa: E402
from reliopt.exec.operators import OPS  # noqa: E402
from reliopt.llm.client import run_parallel  # noqa: E402
from reliopt.quality.features import FEATURES  # noqa: E402
from reliopt.retrieval.bm25 import BM25Index  # noqa: E402

warnings.filterwarnings("ignore")
EV_K = 20
EV_FEATURES = ("ev_count", "ev_first", "ev_top3")


def evidence(index, q, a, question=None):
    na = normalize_answer(a or "")
    hits = index.search(q, k=EV_K) + (index.search(question, k=EV_K) if question else [])
    if not na or not hits:
        return {"ev_count": 0.0, "ev_first": 1.0, "ev_top3": 0.0}
    has = [f" {na} " in f" {normalize_answer(h['title'] + ' ' + h['text'])} " for h in hits]
    first = next((i for i, x in enumerate(has) if x), None)
    return {"ev_count": sum(has) / len(has), "ev_first": 1.0 if first is None else first / len(hits),
            "ev_top3": float(any(has[:3]))}


def add_features(rows, index):
    todo = []
    for r in rows:
        sink = r["plan"]["steps"][-1]["id"]
        for s in r["steps"]:
            if s["kind"] == "TXT":
                todo.append((s, r["question"] if s["id"] == sink else None))
            else:
                s["feat"].update({"ev_count": 0.0, "ev_first": 1.0, "ev_top3": 0.0})
    feats = run_parallel(lambda t: evidence(index, t[0]["q"], t[0]["a0"], t[1]), todo, workers=16)
    for (s, _), f in zip(todo, feats):
        s["feat"].update(f)


def cv_auroc(rows, folds=5):
    from sklearn.metrics import roc_auc_score
    idx = list(range(len(rows)))
    random.Random(0).shuffle(idx)
    ys, ps = [], []
    for f in range(folds):
        test = set(idx[f::folds])
        m = qmod.QualityModel().fit([rows[i] for i in idx if i not in test], OPS)
        st = [s for i in test for s in rows[i]["steps"] if s["label"] in ("correct", "error")]
        ys += [int(s["label"] == "error") for s in st]
        ps += m.eps([s["feat"] for s in st])
    return roc_auc_score(ys, ps)


def use_features(feats):
    qmod.FEATURES = feats
    off.FEATURES = feats


def evaluate(rows, budget):
    idx = list(range(len(rows)))
    random.Random(0).shuffle(idx)
    folds = [idx[i::5] for i in range(5)]
    opts = {v: [None] * len(rows) for v in ("type_rho", "uplift")}
    for f in folds:
        test = set(f)
        train = [rows[i] for i in idx if i not in test]
        qm = qmod.QualityModel().fit(train, OPS)
        im = off.InstanceModels(train)
        for i in f:
            for v in opts:
                opts[v][i] = off.options(rows[i], qm, im, v)
    po = 100 * sum(r["base"]["em"] for r in rows) / len(rows)
    out = {}
    for v, o in opts.items():
        pts = off.curve(rows, o)
        out[v] = {"em_half": off.em_at(pts, budget / 2, po), "em_full": off.em_at(pts, budget, po),
                  "auc": sum(off.em_at(pts, 3000 * k / 100, po) for k in range(101)) / 101, "points": pts}
    return po, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(off.DATASETS))
    args = ap.parse_args()
    base = ROOT / "results" / "e3" / args.tag
    e2 = json.loads((ROOT / "results" / "e2" / args.tag / "test" / "summary.json").read_text())["summary"]
    res = {}
    lines = ["# Offline check: evidence-agreement features (calibration questions, 5-fold CV)", "",
             f"Features from the top-{EV_K} BM25 passages of each TXT step's sub-question (and of the question at a "
             "sink); no model calls. EM in % at verification budgets B_SC/2 and B_SC (at most one verified step per "
             "question); AUC = mean EM over budgets 0..3000.", "",
             "| Dataset | Features | eps AUROC | Allocator | EM@B_SC/2 | EM@B_SC | AUC 0-3000 |", "|---|---|---|---|---|---|---|"]
    for ds in args.datasets:
        rows = [json.loads(l) for l in open(base / "cal" / ds / "questions.jsonl")]
        add_features(rows, BM25Index(ds))
        res[ds] = {}
        for name, feats in (("free", FEATURES), ("free+evidence", tuple(FEATURES) + EV_FEATURES)):
            use_features(feats)
            auroc = cv_auroc(rows)
            po, out = evaluate(rows, e2[ds]["sc5"]["tok"])
            res[ds][name] = {"auroc": auroc, "plan_only": po, **out}
            for v, o in out.items():
                lines.append(f"| {ds} | {name} | {auroc:.3f} | {v} | {o['em_half']:.1f} | {o['em_full']:.1f} | {o['auc']:.1f} |")
        print("\n".join(lines[-4:])); sys.stdout.flush()
    use_features(FEATURES)
    (base / "offline_evidence.json").write_text(json.dumps(res, indent=1))
    (base / "offline_evidence.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
