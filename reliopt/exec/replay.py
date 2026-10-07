"""Execute one question's plan under a verification policy ("one materialisation, offline replay").

Steps run in topological order (plan ids are topologically sorted by validation). At each step the default execution sees
the actual earlier results, then the policy's operator is applied. If an operator changes a step's answer, every later step
that uses it is executed on the new input; because the inputs are part of the prompt, and so of the cache key, only new
input combinations reach the model. Every call a policy would issue is counted, cache hit or not.

A policy is a function step -> operator name; `static(...)` and `at_step(...)` build the two kinds used in e1.
"""
from ..eval.metrics import score
from ..plan import planner
from .operators import Operators, concrete
from .steps import ZERO, StepExecutor, add_cost, resolve, tokens_of


def static(mapping):
    """Type-level policy: {"TXT": op, "CMP": op, "LOG": op} with generic names (NONE / VOTE3 / VOTE5 / ALT / VOTE5+ALT)."""
    return lambda s: concrete(mapping[s["kind"]], s["kind"])


def at_step(step_id, op):
    """Apply `op` at one step only, nothing elsewhere."""
    return lambda s: op if s["id"] == step_id else "NONE"


NO_VERIFY = static({"TXT": "NONE", "CMP": "NONE", "LOG": "NONE"})


class Pipeline:
    def __init__(self, llm, index, seed=17, k=5, txt="short", sink_q=False, cmp_direct=False, untyped=False, strict=True):
        """cmp_direct / untyped / strict=False are the execution-layer ablations of e8 (defaults: the method)."""
        self.llm, self.strict = llm, strict
        self.ex = StepExecutor(llm, index, k=k, seed=seed, txt=txt, sink_q=sink_q, cmp_direct=cmp_direct, untyped=untyped)
        self.ops = Operators(self.ex)

    def plan(self, question):
        return planner.make_plan(self.llm, question, strict=self.strict)

    def run(self, q, plan, policy, keep=False):
        """Returns {"pred", "em", "f1", "acc", "tok", "calls", "steps"}; with keep=True also "_ctx" and "_r0" per step id."""
        steps = plan["steps"]
        sink = steps[-1]["id"]
        answers, resolved = {}, {}
        cost = {"plan": plan["cost"], "exec": dict(ZERO), "verify": dict(ZERO)}
        rows, ctxs, r0s = [], {}, {}
        for s in steps:
            rq = resolve(s["question"], answers)
            ctx = {"question": q["question"], "step": s, "resolved": rq, "is_sink": s["id"] == sink,
                   "inputs": [(d, resolved[d], answers[d]) for d in s["deps"]]}
            r0 = self.ex.default(ctx)
            op = policy(s)
            res = self.ops.apply(op, ctx, r0)
            cost["exec"] = add_cost(cost["exec"], r0["cost"])
            cost["verify"] = add_cost(cost["verify"], res["cost"])
            answers[s["id"]], resolved[s["id"]] = res["answer"], rq
            rows.append({"id": s["id"], "op": op, "a0": r0["answer"], "ans": res["answer"]})
            if keep:
                ctxs[s["id"]], r0s[s["id"]] = ctx, r0
        pred = answers[sink]
        out = {"pred": pred, **score(pred, q["answers"]),
               "tok": {k: tokens_of(v) for k, v in cost.items()}, "calls": {k: v["calls"] for k, v in cost.items()},
               "steps": rows}
        if keep:
            out["_ctx"], out["_r0"] = ctxs, r0s
        return out
