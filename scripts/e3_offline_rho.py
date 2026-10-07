"""Offline check (no model calls): would instance-level repair estimates let the allocator capture more of the oracle
headroom? Run on the materialised calibration questions with 5-fold cross-validation over questions.

Value of applying operator o at step v (the allocator picks, per question, the single best (v, o) with value - lambda * cost > 0):
  A type_rho    the method's decomposition: pi * S * [eps-hat rho+(kind, o) - (1 - eps-hat) rho-(kind, o)]
  B inst_rho    the same, with rho+ / rho- predicted per step from its free features (TXT operators; logistic regression
                on the training folds' wrong steps for rho+, correct steps for rho-; CMP / LOG keep the type-level rates)
  C uplift      no decomposition: P(final answer becomes right) - P(it becomes wrong) after o at v, predicted from the step's
                free features (multinomial logistic regression per TXT operator on the single-step interventions;
                CMP / LOG fall back to A)
  D prior_eps   A with eps-hat replaced by the type prior (reference)
  oracle        the best realised single-step intervention per question (upper bound)
  oracle_step_* the best step with hindsight but a fixed operator per kind (DEEP, or ALT = REGROUND / REROUTE)
Outcome of a choice = the recorded single-step intervention (questions.jsonl "single"), or the plan-only outcome when the
operator leaves the step's answer unchanged. Costs are the measured operator tokens.
Input:  results/e3/<tag>/cal/<ds>/questions.jsonl, results/e2/<tag>/test/summary.json (budgets B_SC, B_SC/2)
Output: results/e3/<tag>/offline_rho.{json,md}
Usage:  python scripts/e3_offline_rho.py --tag llama31-8b
"""
import argparse
import json
import math
import random
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reliopt.exec.operators import OPS  # noqa: E402
from reliopt.quality.features import FEATURES  # noqa: E402
from reliopt.quality.model import QualityModel  # noqa: E402

warnings.filterwarnings("ignore")
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
TXT_OPS = [o for o in OPS["TXT"] if o not in ("NONE", "PROBE")]
LAMS = [0.0] + [10 ** (-8 + 6 * i / 120) for i in range(121)]


def X(steps):
    return [[s["feat"][f] for f in FEATURES] for s in steps]


def lr(Xs, ys):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(Xs, ys)


def outcome(r, s, op):
    res = r["single"].get(f"{s['id']}:{op}")
    return res["em"] if res else r["base"]["em"]


class InstanceModels:
    """rho+ / rho- per TXT operator from free features, and the direct uplift model."""

    def __init__(self, rows):
        self.rp, self.rm, self.up = {}, {}, {}
        txt = [(r, s) for r in rows for s in r["steps"] if s["kind"] == "TXT"]
        for op in TXT_OPS:
            wrong = [s for _, s in txt if s["label"] == "error" and s["ops"].get(op, {}).get("label") in ("correct", "error")]
            right = [s for _, s in txt if s["label"] == "correct" and s["ops"].get(op, {}).get("label") in ("correct", "error")]
            yw = [int(s["ops"][op]["label"] == "correct") for s in wrong]
            yr = [int(s["ops"][op]["label"] == "error") for s in right]
            if 10 <= sum(yw) <= len(yw) - 10:
                self.rp[op] = lr(X(wrong), yw)
            if 10 <= sum(yr) <= len(yr) - 10:
                self.rm[op] = lr(X(right), yr)
            pairs = [(s, outcome(r, s, op) - r["base"]["em"]) for r, s in txt if op in s["ops"]]
            ys = [int(round(d)) + 1 for _, d in pairs]   # 0: becomes wrong, 1: unchanged, 2: becomes right
            if len(set(ys)) == 3 and min(ys.count(c) for c in (0, 2)) >= 5:
                self.up[op] = lr(X([s for s, _ in pairs]), ys)

    @staticmethod
    def p1(m, s):
        return float(m.predict_proba(X([s]))[0, 1])

    def uplift(self, op, s):
        if op not in self.up:
            return None
        p = self.up[op].predict_proba(X([s]))[0]
        cls = list(self.up[op].classes_)
        return p[cls.index(2)] - p[cls.index(0)]


def options(r, qm, im, variant):
    """(value, cost, realised em) of every (step, operator) of one question."""
    steps, plan = r["steps"], r["plan"]["steps"]
    eps = {s["id"]: (qm.prior(s["kind"]) if variant == "prior_eps" else e)
           for s, e in zip(steps, qm.eps([s["feat"] for s in steps]))}
    pis = qm.pis(plan)
    out = []
    for s in steps:
        rest = math.prod(1 - eps[u["id"]] * pis[u["id"]] for u in steps if u["id"] != s["id"])
        w = pis[s["id"]] * rest
        for op in OPS[s["kind"]]:
            if op in ("NONE", "PROBE") or op not in s["ops"]:
                continue
            rp, rm = qm.rho.get((s["kind"], op), (0.0, 0.0))
            e = eps[s["id"]]
            if variant == "uplift" and s["kind"] == "TXT" and im.uplift(op, s) is not None:
                val = im.uplift(op, s)
            else:
                if variant == "inst_rho" and s["kind"] == "TXT":
                    rp = im.p1(im.rp[op], s) if op in im.rp else rp
                    rm = im.p1(im.rm[op], s) if op in im.rm else rm
                val = w * (e * rp - (1 - e) * rm)
            out.append((val, s["ops"][op]["tok"], outcome(r, s, op)))
    return out


def curve(rows, opts):
    pts = []
    for lam in LAMS:
        em = tok = 0.0
        for r, o in zip(rows, opts):
            best = max(o, key=lambda x: x[0] - lam * x[1], default=None)
            if best and best[0] - lam * best[1] > 0:
                em, tok = em + best[2], tok + best[1]
            else:
                em += r["base"]["em"]
        pts.append((tok / len(rows), 100 * em / len(rows)))
    return sorted(set(pts))


def oracle(rows):
    pts = []
    for lam in LAMS + [10 ** (-4 + i / 10) for i in range(20)]:
        em = tok = 0.0
        for r in rows:
            acts = [(r["base"]["em"], 0)] + [(outcome(r, s, op), s["ops"][op]["tok"]) for s in r["steps"]
                                            for op in s["ops"] if op != "PROBE"]
            best = max(acts, key=lambda a: (a[0] - lam * a[1], -a[1]))
            em, tok = em + best[0], tok + best[1]
        pts.append((tok / len(rows), 100 * em / len(rows)))
    return sorted(set(pts))


def oracle_step(rows, ops=None):
    """Oracle over the step only: the operator is fixed per kind (default: the strongest, DEEP), the best step chosen
    with hindsight. Separates the value of picking the right step from picking a lucky operator among many."""
    ops = ops or {"TXT": "DEEPGROUND", "CMP": "VOTE5+REROUTE", "LOG": "VOTE5+REROUTE"}
    pts = []
    for lam in LAMS + [10 ** (-4 + i / 10) for i in range(20)]:
        em = tok = 0.0
        for r in rows:
            acts = [(r["base"]["em"], 0)] + [(outcome(r, s, ops[s["kind"]]), s["ops"][ops[s["kind"]]]["tok"])
                                            for s in r["steps"] if ops[s["kind"]] in s["ops"]]
            best = max(acts, key=lambda a: (a[0] - lam * a[1], -a[1]))
            em, tok = em + best[0], tok + best[1]
        pts.append((tok / len(rows), 100 * em / len(rows)))
    return sorted(set(pts))


def em_at(pts, budget, base):
    """Upper envelope interpolation: best mixture of two operating points within the budget."""
    pts = [(0.0, base)] + [p for p in pts]
    best = max((e for c, e in pts if c <= budget), default=base)
    for c0, e0 in pts:
        for c1, e1 in pts:
            if c0 <= budget < c1:
                best = max(best, e0 + (e1 - e0) * (budget - c0) / (c1 - c0))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    base = ROOT / "results" / "e3" / args.tag
    e2 = json.loads((ROOT / "results" / "e2" / args.tag / "test" / "summary.json").read_text())["summary"]
    res, lines = {}, ["# Offline check: instance-level repair estimates (calibration questions, 5-fold CV, at most one "
                      "verified step per question)", "",
                      "EM in %; budgets are verification tokens per question (B_SC of the dataset and half of it); "
                      "AUC = mean EM over budgets 0..3000.", "",
                      "| Dataset | Plan-only | Variant | EM@B_SC/2 | EM@B_SC | AUC 0-3000 |", "|---|---|---|---|---|---|"]
    for ds in args.datasets:
        rows = [json.loads(l) for l in open(base / "cal" / ds / "questions.jsonl")]
        idx = list(range(len(rows)))
        random.Random(0).shuffle(idx)
        folds = [idx[i::args.folds] for i in range(args.folds)]
        opts = {v: [None] * len(rows) for v in ("type_rho", "inst_rho", "uplift", "prior_eps")}
        for f in folds:
            test = set(f)
            train = [rows[i] for i in idx if i not in test]
            qm = QualityModel().fit(train, OPS)
            im = InstanceModels(train)
            for i in f:
                for v in opts:
                    opts[v][i] = options(rows[i], qm, im, v)
        po = 100 * sum(r["base"]["em"] for r in rows) / len(rows)
        b = e2[ds]["sc5"]["tok"]
        res[ds] = {"plan_only": po, "budgets": [b / 2, b], "curves": {}}
        cur = {v: curve(rows, o) for v, o in opts.items()}
        cur["oracle"] = oracle(rows)
        cur["oracle_step_deep"] = oracle_step(rows)
        cur["oracle_step_alt"] = oracle_step(rows, {"TXT": "REGROUND", "CMP": "REROUTE", "LOG": "REROUTE"})
        for v, pts in cur.items():
            row = {"em_half": em_at(pts, b / 2, po), "em_full": em_at(pts, b, po),
                   "auc": sum(em_at(pts, 3000 * k / 100, po) for k in range(101)) / 101}
            res[ds]["curves"][v] = {"points": pts, **row}
            lines.append(f"| {ds} | {po:.1f} | {v} | {row['em_half']:.1f} | {row['em_full']:.1f} | {row['auc']:.1f} |")
        print("\n".join(lines[-7:])); sys.stdout.flush()
    (base / "offline_rho.json").write_text(json.dumps(res, indent=1))
    (base / "offline_rho.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
