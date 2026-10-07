"""e1 side check: how much of the oracle headroom does a deployable, learned allocator reach?

A conservative single-step version of the method, computed offline from an e1 round (no new model calls):
  - eps_hat: logistic regression on the free features (standardised), trained with 5-fold cross-validation over questions,
    so every question is scored by a model that never saw its labels; trained on steps whose label is decidable;
  - rho+/-(kind, op) and pi(kind, position) are counted on the training folds (pi = P(final wrong | step error) among
    questions whose other steps are all correct, Beta(1,1)-smoothed);
  - per question, the step-operator pair with the largest  pi * [eps*rho+ - (1-eps)*rho-] - lambda * cost  is applied if
    positive (at most one per question); its outcome is the recorded single-step intervention (or the plan-only outcome when
    the operator leaves the answer unchanged).
Sweeping lambda gives an EM / verification-token curve, compared with type-level policies (in-sample best at each cost,
favourable to the baseline), Verify-All and the single-step oracle.
Input:  results/e1/<run>/<ds>/questions.jsonl
Output: results/e1/<run>/allocsim.json, printed table
Usage:  python scripts/e1_allocsim.py --run r3
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reliopt.quality.features import FEATURES  # noqa: E402

OPS = {"TXT": ("VOTE3", "VOTE5", "REGROUND", "VOTE5+REGROUND", "DEEPGROUND"),
       "CMP": ("VOTE3", "VOTE5", "REROUTE", "VOTE5+REROUTE"), "LOG": ("VOTE3", "VOTE5", "REROUTE", "VOTE5+REROUTE")}


def fit_predict(train, test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = [[s["feat"][f] for f in FEATURES] for s in train]
    y = [int(s["label"] == "error") for s in train]
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)
    return m.predict_proba([[s["feat"][f] for f in FEATURES] for s in test])[:, 1]


def position(r, s):
    return "sink" if s["id"] == r["plan"]["steps"][-1]["id"] else "inner"


def tables(rows):
    """rho+/- per (kind, op) and pi per (kind, position) from the training questions."""
    rho = defaultdict(lambda: [1, 2, 1, 2])   # fixed+1, n_err+2, damaged+1, n_ok+2 (Beta(1,1))
    pi = defaultdict(lambda: [1, 2])
    for r in rows:
        labs = {s["id"]: s["label"] for s in r["steps"]}
        for s in r["steps"]:
            if s["label"] == "undet":
                continue
            for op in OPS[s["kind"]]:
                o = s["ops"].get(op)
                if not o or o["label"] == "undet":
                    continue
                c = rho[(s["kind"], op)]
                if s["label"] == "error":
                    c[0] += o["label"] == "correct"; c[1] += 1
                else:
                    c[2] += o["label"] == "error"; c[3] += 1
            others_ok = all(l == "correct" for i, l in labs.items() if i != s["id"])
            if s["label"] == "error" and others_ok:
                p = pi[(s["kind"], position(r, s))]
                p[0] += r["base"]["em"] == 0; p[1] += 1
    rho_t = {k: (v[0] / v[1], v[2] / v[3]) for k, v in rho.items()}
    pi_t = {k: v[0] / v[1] for k, v in pi.items()}
    return rho_t, pi_t


def options(r, eps, rho_t, pi_t):
    """(benefit estimate, cost, realised em) for every step-operator pair of one question."""
    out = []
    for s, e in zip(r["steps"], eps):
        p = pi_t.get((s["kind"], position(r, s)), 0.5)
        for op in OPS[s["kind"]]:
            if op not in s["ops"]:
                continue
            rp, rm = rho_t.get((s["kind"], op), (0.0, 0.0))
            gain = p * (e * rp - (1 - e) * rm)
            res = r["single"].get(f"{s['id']}:{op}")
            em = res["em"] if res else r["base"]["em"]
            out.append((gain, s["ops"][op]["tok"], em))
    return out


def curve(rows, opts):
    lams = [0.0] + [10 ** (-7 + 6 * i / 300) for i in range(301)]
    pts = []
    for lam in lams:
        em = tok = 0.0
        for r, o in zip(rows, opts):
            best = max(o, key=lambda x: x[0] - lam * x[1], default=None)
            if best and best[0] - lam * best[1] > 0:
                em += best[2]; tok += best[1]
            else:
                em += r["base"]["em"]
        pts.append((tok / len(rows), 100 * em / len(rows)))
    return sorted(set((round(t, 1), round(e, 2)) for t, e in pts))


def best_at(points, cost):
    ok = [e for t, e in points if t <= cost + 1e-9]
    return max(ok) if ok else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="r3")
    ap.add_argument("--datasets", nargs="+", default=["2wikimultihopqa", "musique"])
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    summary = json.loads((ROOT / "results" / "e1" / args.run / "summary.json").read_text())
    out = {}
    for ds in args.datasets:
        rows = [json.loads(l) for l in open(ROOT / "results" / "e1" / args.run / ds / "questions.jsonl")]
        idx = list(range(len(rows)))
        random.Random(0).shuffle(idx)
        folds = [idx[i::args.folds] for i in range(args.folds)]
        opts = [None] * len(rows)
        for f in folds:
            test = set(f)
            train_rows = [rows[i] for i in idx if i not in test]
            rho_t, pi_t = tables(train_rows)
            train_steps = [s for r in train_rows for s in r["steps"] if s["label"] != "undet"]
            for i in f:
                eps = fit_predict(train_steps, rows[i]["steps"])
                opts[i] = options(rows[i], eps, rho_t, pi_t)
        pts = curve(rows, opts)
        s = summary["datasets"][ds]
        grid = [(v["verify_tok"], v["em"]) for v in s["grid"].values()]
        po = s["plan_only"]["em"]
        orc = [(p["verify_tok"], p["em"]) for p in s["oracle_single"]]
        res = {"plan_only": po, "learned_curve": pts}
        print(f"\n{ds}: plan-only EM {po:.1f}; learned single-step allocator (5-fold CV)")
        print(f"{'verify tok/question':>20} {'learned':>8} {'type-level best':>16} {'oracle':>7}")
        for c in (50, 100, 200, 400, 800, 1600, 3200):
            row = (c, best_at(pts, c), best_at(grid, c), best_at(orc, c))
            res.setdefault("table", []).append(row)
            print(f"{c:>20} {row[1]:>8.1f} {row[2]:>16.1f} {row[3]:>7.1f}")
        top = max(pts, key=lambda p: (p[1], -p[0]))
        res["learned_best"] = top
        va = s["policies"].get("verify_all_deep") or s["policies"]["verify_all"]
        print(f"learned best: EM {top[1]:.1f} at {top[0]:.0f} verify tokens/question; "
              f"Verify-All (deep) EM {va['em']:.1f} at {va['verify_tok']:.0f}")
        out[ds] = res
    (ROOT / "results" / "e1" / args.run / "allocsim.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
