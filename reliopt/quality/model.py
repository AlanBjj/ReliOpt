"""Quality model, fitted on materialised calibration questions (rows of scripts/e1_pilot.py's
process(): plan, base run, and per step the label, free features and every operator applied on the plan-only inputs).

  eps(phi)            P(local error | free features): L2 logistic regression (C chosen by 5-fold CV) + isotonic calibration
  eps_probe(phi, b)   the same with the PROBE outcome b (1 = the probe sample disagrees with the default answer) as a feature
  p_disagree(phi)     P(b = 1 | phi), a second light logistic regression on the same features
  rho[kind, op]       (rho+, rho-) = (P(op's answer correct | default wrong), P(op's answer wrong | default correct)),
                      Beta(1,1)-smoothed counts over decidable steps
  tau[edge type]      P(an error of the parent makes the child's result wrong), edge type = "<parent kind>><child kind>";
                      maximum likelihood over questions with exactly one wrong step (its descendants are undecidable
                      by construction; every other step decidable and correct), where the wrong step's
                      pi = 1 - prod_paths (1 - prod tau) must explain the final outcome
  eps_bar[kind]       type prior for steps not yet executed (calibration error frequency)
  cost[kind, op]      mean verification tokens of op on that kind of step (the paper's c-hat), tokens actually counted
Gold annotations only enter through the calibration labels; nothing here looks at a test question's gold.
"""
import math
from collections import defaultdict

from .features import FEATURES

EDGE_INIT = 0.9


def _X(steps, extra=None):
    return [[s["feat"][f] for f in FEATURES] + ([extra(s)] if extra else []) for s in steps]


def _fit_lr(X, y, calibrate=True):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    base = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0], cv=5, max_iter=5000,
                                                                scoring="neg_log_loss"))
    if not calibrate or min(sum(y), len(y) - sum(y)) < 10:
        return base.fit(X, y)
    return CalibratedClassifierCV(base, method="isotonic", cv=5).fit(X, y)


def paths_to_sink(steps):
    """For each step id, the list of paths to the sink, each a list of edges (parent kind, child kind)."""
    kind = {s["id"]: s["kind"] for s in steps}
    children = defaultdict(list)
    for s in steps:
        for d in s["deps"]:
            children[d].append(s["id"])
    sink = steps[-1]["id"]
    memo = {}

    def go(v):
        if v == sink:
            return [[]]
        if v not in memo:
            memo[v] = [[(kind[v], kind[c])] + p for c in children[v] for p in go(c)]
        return memo[v]
    return {s["id"]: go(s["id"]) for s in steps}


def descendants(steps, v):
    out, frontier = set(), {v}
    while frontier:
        frontier = {s["id"] for s in steps if set(s["deps"]) & frontier} - out
        out |= frontier
    return out


def edge_key(e):
    return f"{e[0]}>{e[1]}"


def pi_from_paths(paths, tau):
    """1 - prod over paths of (1 - prod of tau along the path); a step with no path to the sink has pi = 0."""
    if paths == [[]]:
        return 1.0
    miss = 1.0
    for p in paths:
        miss *= 1.0 - math.prod(tau.get(edge_key(e), EDGE_INIT) for e in p)
    return 1.0 - miss


class QualityModel:
    def __init__(self, calibrate=True, estimator="logreg"):
        self.calibrate, self.estimator = calibrate, estimator

    # ---- fitting ------------------------------------------------------------------------------------------------------
    def fit(self, rows, ops_by_kind):
        steps = [s for r in rows for s in r["steps"] if s["label"] in ("correct", "error")]
        y = [int(s["label"] == "error") for s in steps]
        self.n_steps, self.n_err = len(steps), sum(y)
        self.m_eps = self._fit(_X(steps), y)
        probed = [s for s in steps if "PROBE" in s["ops"] and "agree" in s["ops"]["PROBE"]]
        dis = lambda s: float(not s["ops"]["PROBE"]["agree"])
        self.m_eps_probe = self._fit(_X(probed, dis), [int(s["label"] == "error") for s in probed])
        all_probed = [s for r in rows for s in r["steps"] if "PROBE" in s["ops"] and "agree" in s["ops"]["PROBE"]]
        self.m_disagree = _fit_lr(_X(all_probed), [int(dis(s)) for s in all_probed], calibrate=False)
        cnt = defaultdict(lambda: [0, 0])
        for s in steps:
            cnt[s["kind"]][0] += s["label"] == "error"; cnt[s["kind"]][1] += 1
        self.eps_bar = {k: (e + 1) / (n + 2) for k, (e, n) in cnt.items()}
        rho = defaultdict(lambda: [1, 2, 1, 2])   # Beta(1,1): fixed + 1 / n_err + 2, damaged + 1 / n_ok + 2
        cost = defaultdict(list)
        for r in rows:
            for s in r["steps"]:
                for op, o in s["ops"].items():
                    cost[(s["kind"], op)].append(o["tok"])
                    if op == "PROBE" or s["label"] not in ("correct", "error") or o.get("label") not in ("correct", "error"):
                        continue
                    c = rho[(s["kind"], op)]
                    if s["label"] == "error":
                        c[0] += o["label"] == "correct"; c[1] += 1
                    else:
                        c[2] += o["label"] == "error"; c[3] += 1
        self.rho = {k: (v[0] / v[1], v[2] / v[3]) for k, v in rho.items()}
        self.rho_n = {k: (v[1] - 2, v[3] - 2) for k, v in rho.items()}
        self.cost = {k: sum(v) / len(v) for k, v in cost.items()}
        self.ops_by_kind = {k: [o for o in ops if o != "NONE"] for k, ops in ops_by_kind.items()}
        self.tau = self._fit_tau(rows)
        return self

    def _fit(self, X, y):
        if self.estimator == "gbdt":
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.ensemble import HistGradientBoostingClassifier
            base = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200)
            return CalibratedClassifierCV(base, method="isotonic", cv=5).fit(X, y) if self.calibrate else base.fit(X, y)
        return _fit_lr(X, y, calibrate=self.calibrate)

    def _fit_tau(self, rows):
        """MLE of tau per edge type from single-error questions (others decidable and correct)."""
        from scipy.optimize import minimize
        data = []   # (paths of the wrong step, final wrong?)
        for r in rows:
            labs = {s["id"]: s["label"] for s in r["steps"]}
            errs = [i for i, l in labs.items() if l == "error"]
            if len(errs) != 1:
                continue
            # descendants of the wrong step are "undet" by construction (labels need correct parents); every other step
            # must be decidable and correct
            desc = descendants(r["plan"]["steps"], errs[0])
            if any(l != "correct" for i, l in labs.items() if i != errs[0] and i not in desc):
                continue
            paths = paths_to_sink(r["plan"]["steps"])[errs[0]]
            if paths == [[]]:
                continue   # the sink itself: pi = 1 by definition
            data.append((paths, int(r["base"]["em"] == 0)))
        keys = sorted({edge_key(e) for paths, _ in data for p in paths for e in p})
        self.tau_n = defaultdict(int)
        for paths, _ in data:
            for k in {edge_key(e) for p in paths for e in p}:
                self.tau_n[k] += 1
        if not keys:
            return {}

        def nll(z):
            tau = {k: 1 / (1 + math.exp(-x)) for k, x in zip(keys, z)}
            out = 0.0
            for paths, wrong in data:
                p = min(max(pi_from_paths(paths, tau), 1e-6), 1 - 1e-6)
                out -= math.log(p if wrong else 1 - p)
            # Beta(2,1)-like pull towards "errors propagate" for rarely seen edge types
            return out - sum(math.log(1 / (1 + math.exp(-x))) for x in z)
        z0 = [math.log(EDGE_INIT / (1 - EDGE_INIT))] * len(keys)
        res = minimize(nll, z0, method="L-BFGS-B", bounds=[(-8, 8)] * len(keys))
        return {k: 1 / (1 + math.exp(-x)) for k, x in zip(keys, res.x)}

    # ---- prediction ---------------------------------------------------------------------------------------------------
    def eps(self, feats):
        return list(self.m_eps.predict_proba(_X([{"feat": f} for f in feats]))[:, 1])

    def eps_probe(self, feat, disagree):
        return float(self.m_eps_probe.predict_proba(_X([{"feat": feat}], lambda s: float(disagree)))[0, 1])

    def p_disagree(self, feat):
        return float(self.m_disagree.predict_proba(_X([{"feat": feat}]))[0, 1])

    def prior(self, kind):
        """Type prior eps-bar; Beta(1,1) mean 0.5 for a kind with no decidable calibration step (e.g. CMP on MuSiQue)."""
        return self.eps_bar.get(kind, 0.5)

    def pis(self, plan_steps, propagate=True):
        if not propagate:
            return {s["id"]: 1.0 for s in plan_steps}
        return {i: pi_from_paths(p, self.tau) for i, p in paths_to_sink(plan_steps).items()}

    def summary(self):
        return {"n_steps": self.n_steps, "n_err": self.n_err, "eps_bar": self.eps_bar,
                "rho": {f"{k[0]}:{k[1]}": v for k, v in self.rho.items()},
                "rho_n": {f"{k[0]}:{k[1]}": v for k, v in self.rho_n.items()},
                "tau": self.tau, "tau_n": dict(self.tau_n),
                "cost": {f"{k[0]}:{k[1]}": v for k, v in self.cost.items()}}
