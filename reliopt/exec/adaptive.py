"""Adaptive execution with a price lambda and the Question-Level allocator of §5.6.

run_adaptive: steps in topological order; each step is executed by default, its free features give eps-hat, and with
  pi-hat (edge-type propagation) and S-hat_{-v} (other steps: posterior error after their operator if executed, type prior
  if not) every operator's net value  N(v, o) = pi * S * [eps rho+ - (1 - eps) rho-] - lambda * c(kind, o)  is computed.
  PROBE runs when its value of information exceeds its priced cost; eps is then re-estimated with the probe outcome.
  The operator with the largest positive N is applied (NONE when none is positive), and the step's posterior error
  eps (1 - rho+) + (1 - eps) rho- enters S-hat of later steps. A changed answer re-executes the steps that use it (their
  default executions see the new input).
  Variants: probe=False (w/o escalation: operators chosen from the free features only), propagate=False
  (w/o propagation: pi = S = 1), prior_eps=True (w/o instance estimates: eps = the type prior).
Cost accounting: every call is counted (cache hits included). PROBE reads the first pool sample; a VOTE after it reuses
  that sample, so it is charged its own request minus that sample's completion (prompt paid again: a second request).
run_question_level: execute the plan without verification, form the question-level error estimate eps_q = mean eps-hat
  of its steps, and give every step the same type-level operator g (NONE / VOTE3 / VOTE5 / ALT / VOTE5+ALT / DEEP) that
  maximises sum_v pi S [eps_q rho+ - (1 - eps_q) rho-] - lambda * sum_v c; the question's budget follows its estimate, and
  is spread evenly over its steps.
"""
import math

from ..eval.metrics import score
from ..plan.planner import structure
from ..quality.features import step_features
from .operators import GENERIC, OPS, concrete
from .replay import NO_VERIFY, static
from .steps import ZERO, add_cost, resolve, tokens_of

VOTING = ("VOTE3", "VOTE5", "VOTE5+REGROUND", "VOTE5+REROUTE")


class Allocator:
    def __init__(self, qm, lam, probe=True, propagate=True, prior_eps=False):
        self.qm, self.lam, self.probe, self.propagate, self.prior_eps = qm, lam, probe, propagate, prior_eps

    def c(self, kind, op):
        return self.qm.cost.get((kind, op), 0.0) if op != "NONE" else 0.0

    def rho(self, kind, op):
        return self.qm.rho.get((kind, op), (0.0, 0.0)) if op != "NONE" else (0.0, 0.0)

    def best(self, kind, eps, weight):
        """(net value, operator) maximising pi S [eps rho+ - (1-eps) rho-] - lambda c over the step's operators."""
        out = (0.0, "NONE")
        for op in OPS[kind]:
            if op in ("NONE", "PROBE"):
                continue
            rp, rm = self.rho(kind, op)
            v = weight * (eps * rp - (1 - eps) * rm) - self.lam * self.c(kind, op)
            if v > out[0]:
                out = (v, op)
        return out


def run_adaptive(pipe, q, plan, alloc):
    qm = alloc.qm
    steps = plan["steps"]
    sink = steps[-1]["id"]
    struct = structure(steps)
    pis = qm.pis(steps, propagate=alloc.propagate)
    kinds = {s["id"]: s["kind"] for s in steps}
    answers, resolved, post = {}, {}, {}
    cost = {"plan": plan["cost"], "exec": dict(ZERO), "verify": dict(ZERO)}
    trace = []
    for s in steps:
        sid, kind = s["id"], s["kind"]
        rq = resolve(s["question"], answers)
        ctx = {"question": q["question"], "step": s, "resolved": rq, "is_sink": sid == sink,
               "inputs": [(d, resolved[d], answers[d]) for d in s["deps"]]}
        r0 = pipe.ex.default(ctx)
        cost["exec"] = add_cost(cost["exec"], r0["cost"])
        feat = step_features(struct[sid], ctx, r0)
        eps = qm.prior(kind) if alloc.prior_eps else qm.eps([feat])[0]
        if alloc.propagate:
            rest = math.prod(1 - (post[u] if u in post else qm.prior(kinds[u])) * pis[u]
                             for u in kinds if u != sid)
        else:
            rest = 1.0
        weight = pis[sid] * rest
        probed, b = False, None
        if alloc.probe and not alloc.prior_eps:
            pd = qm.p_disagree(feat)
            e_dis, e_agr = qm.eps_probe(feat, 1), qm.eps_probe(feat, 0)
            voi = (pd * alloc.best(kind, e_dis, weight)[0] + (1 - pd) * alloc.best(kind, e_agr, weight)[0]
                   - alloc.best(kind, eps, weight)[0])
            if voi > alloc.lam * alloc.c(kind, "PROBE"):
                pr = pipe.ops.apply("PROBE", ctx, r0)
                cost["verify"] = add_cost(cost["verify"], pr["cost"])
                probed, b = True, not pr["info"]["agree"]
                eps = e_dis if b else e_agr
        _, op = alloc.best(kind, eps, weight)
        if op == "NONE":
            ans = r0["answer"]
        else:
            res = pipe.ops.apply(op, ctx, r0)
            c = dict(res["cost"])
            if probed and op in VOTING:   # the probe sample is reused by the vote
                c["completion"] -= pipe.ex.pool(ctx)["completions"][0]
            cost["verify"] = add_cost(cost["verify"], c)
            ans = res["answer"]
        rp, rm = alloc.rho(kind, op)
        post[sid] = eps * (1 - rp) + (1 - eps) * rm
        answers[sid], resolved[sid] = ans, rq
        trace.append({"id": sid, "kind": kind, "eps": round(eps, 4), "pi": round(pis[sid], 4), "rest": round(rest, 4),
                      "probe": b, "op": op, "a0": r0["answer"], "ans": ans})
    pred = answers[sink]
    return {"pred": pred, **score(pred, q["answers"]), "tok": {k: tokens_of(v) for k, v in cost.items()},
            "calls": {k: v["calls"] for k, v in cost.items()}, "steps": trace}


def run_question_level(pipe, q, plan, qm, lam):
    base = pipe.run(q, plan, NO_VERIFY, keep=True)
    steps = plan["steps"]
    struct = structure(steps)
    feats = [step_features(struct[s["id"]], base["_ctx"][s["id"]], base["_r0"][s["id"]]) for s in steps]
    eps_q = sum(qm.eps(feats)) / len(steps)
    pis = qm.pis(steps)
    rest = {s["id"]: math.prod(1 - eps_q * pis[u["id"]] for u in steps if u["id"] != s["id"]) for s in steps}
    best, best_v = "NONE", 0.0
    for g in GENERIC[1:]:
        v = 0.0
        for s in steps:
            op = concrete(g, s["kind"])
            rp, rm = qm.rho.get((s["kind"], op), (0.0, 0.0))
            v += pis[s["id"]] * rest[s["id"]] * (eps_q * rp - (1 - eps_q) * rm) - lam * qm.cost.get((s["kind"], op), 0.0)
        if v > best_v:
            best, best_v = g, v
    out = base if best == "NONE" else pipe.run(q, plan, static({"TXT": best, "CMP": best, "LOG": best}))
    out = {k: v for k, v in out.items() if not k.startswith("_")}
    out["policy"], out["eps_q"] = best, round(eps_q, 4)
    return out
