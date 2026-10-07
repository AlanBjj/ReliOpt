"""Planner: the frozen model decomposes a question into a JSON plan of typed, dependent steps.

A plan is {"steps": [{"id", "kind", "question", "deps"}], "fallback", "attempts", "error", "raw", "cost"}.
Validation: 1-6 steps with ids 1..n in order; kind in TXT/CMP/LOG; every "#k" reference and every depends_on entry names an
earlier step; CMP and LOG steps depend on something; the last step is the only sink (every other step is used later);
from round 4, the last step asks yes/no only if the question does. Normalisation (round 4): a LOG step that uses one earlier
result and is not a yes/no question asks for a fact about that result, so it becomes a TXT step (LOG does not retrieve).
An invalid plan is sent back once with the validation error (greedy, so the retry is deterministic and cached); if the
second plan is also invalid the question falls back to a single TXT step on the original question.
The plan is a deterministic function of (backbone, prompt, question), so the LLM cache makes it shared by every policy.
"""
import json
import re

from ..exec.steps import is_yesno

MAX_STEPS = 6
MAX_PLAN_TOKENS = 384
KINDS = ("TXT", "CMP", "LOG")
REF = re.compile(r"#(\d+)")

SYSTEM = """Decompose the question into a plan of at most 6 steps.
Each step has: id, kind (TXT | CMP | LOG), question, depends_on.
A step may refer to earlier results as #<id>. Exactly one final step: the last step gives the answer to the question, and every other step must be used by a later step.
TXT: find one fact with a search. CMP: compare or compute over earlier results.
LOG: combine earlier results, judge yes/no, or state the final answer.
Each TXT step asks for exactly one fact about one entity that is already named. If the entity is only described (for example "the director of film X"), find it in its own earlier step first.
The step that gives the answer to the question is the last step: never add a step after it. Use LOG only for a yes/no judgment or to combine several results.
Return JSON only, in the form {"steps": [{"id": 1, "kind": "TXT", "question": "...", "depends_on": []}, ...]}. Do not answer any step."""

# Fictional entities, so the examples cannot leak facts from the benchmarks.
FEWSHOT = [
    ("In which city was the director of the film Northern Shore born?",
     [(1, "TXT", "Who directed the film Northern Shore?", []),
      (2, "TXT", "In which city was #1 born?", [1])]),
    ("Which company was founded earlier, Arlen Motors or Bexley Foods?",
     [(1, "TXT", "When was Arlen Motors founded?", []),
      (2, "TXT", "When was Bexley Foods founded?", []),
      (3, "CMP", "Which was founded earlier, Arlen Motors (founded #1) or Bexley Foods (founded #2)?", [1, 2])]),
    ("Which film has the director who was born earlier, Northern Shore or Copper Valley?",
     [(1, "TXT", "Who directed the film Northern Shore?", []),
      (2, "TXT", "Who directed the film Copper Valley?", []),
      (3, "TXT", "When was #1 born?", [1]),
      (4, "TXT", "When was #2 born?", [2]),
      (5, "CMP", "Which film has the director who was born earlier, Northern Shore (director born #3) or Copper Valley "
                 "(director born #4)?", [3, 4])]),
    ("Are the authors of the novels Grey Harbor and The Tin Orchard of the same nationality?",
     [(1, "TXT", "Who wrote the novel Grey Harbor?", []),
      (2, "TXT", "Who wrote the novel The Tin Orchard?", []),
      (3, "TXT", "What is the nationality of #1?", [1]),
      (4, "TXT", "What is the nationality of #2?", [2]),
      (5, "LOG", "Are the nationalities #3 and #4 the same?", [3, 4])]),
    ("Who is the father-in-law of Mira Holt?",
     [(1, "TXT", "Who is the spouse of Mira Holt?", []),
      (2, "TXT", "Who is the father of #1?", [1])]),
    ("What is the capital of the country where the composer of the opera Silver Lantern was born?",
     [(1, "TXT", "Who composed the opera Silver Lantern?", []),
      (2, "TXT", "In which country was #1 born?", [1]),
      (3, "TXT", "What is the capital of #2?", [2])]),
    # round 4: a choice and an intersection question, whose last step asks for the answer itself, not for yes/no
    ("Which is a jazz record label, Halden Records or Moorgate Sound?",
     [(1, "TXT", "What kind of music does Halden Records release?", []),
      (2, "TXT", "What kind of music does Moorgate Sound release?", []),
      (3, "LOG", "Which is a jazz record label, Halden Records (releases #1) or Moorgate Sound (releases #2)?", [1, 2])]),
    ("Both Ivo Brandt and Kasia Lund played for which football club?",
     [(1, "TXT", "Which football clubs did Ivo Brandt play for?", []),
      (2, "TXT", "Which football clubs did Kasia Lund play for?", []),
      (3, "LOG", "Which football club appears in both #1 and #2?", [1, 2])]),
]


def _plan_json(steps):
    return json.dumps({"steps": [{"id": i, "kind": k, "question": q, "depends_on": d} for i, k, q, d in steps]})


def messages(question):
    msgs = [{"role": "system", "content": SYSTEM}]
    for q, steps in FEWSHOT:
        msgs += [{"role": "user", "content": f"Question: {q}"}, {"role": "assistant", "content": _plan_json(steps)}]
    return msgs + [{"role": "user", "content": f"Question: {question}"}]


def _extract_json(text):
    m = re.search(r"```(?:json)?\s*(.+?)```", text, flags=re.S)
    if m:
        text = m.group(1)
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON object found")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    return json.loads(text[start:end + 1])


def _as_int(x):
    if isinstance(x, int):
        return x
    m = re.fullmatch(r"\s*#?(\d+)\s*", str(x))
    if not m:
        raise ValueError(f"bad step id {x!r}")
    return int(m.group(1))


def parse(text, question=None, normalize=True):
    """Return (steps, None) for a valid plan, or (None, reason) for an invalid one. With `question`, a plan whose last
    step asks yes/no is invalid unless the question itself does (round 4: such plans answered "yes" to "which" questions).
    normalize=False (e8 ablation) keeps a single-input non-yes/no LOG step as LOG instead of turning it into a TXT look-up."""
    try:
        obj = _extract_json(text)
    except Exception as e:
        return None, f"the reply is not valid JSON ({type(e).__name__})"
    raw = obj.get("steps") if isinstance(obj, dict) else obj
    if not isinstance(raw, list) or not raw:
        return None, 'the JSON must contain a non-empty list "steps"'
    if len(raw) > MAX_STEPS:
        return None, f"the plan has {len(raw)} steps; at most {MAX_STEPS} are allowed"
    steps = []
    for pos, s in enumerate(raw, 1):
        if not isinstance(s, dict):
            return None, f"step {pos} is not an object"
        try:
            sid = _as_int(s.get("id", pos))
            dec = [_as_int(d) for d in (s.get("depends_on") or [])]
        except ValueError as e:
            return None, str(e)
        if sid != pos:
            return None, f"step ids must be 1, 2, ... in order; step {pos} has id {sid}"
        kind = str(s.get("kind", "")).strip().upper()
        if kind not in KINDS:
            return None, f"step {sid} has kind {s.get('kind')!r}; use TXT, CMP or LOG"
        q = str(s.get("question", "")).strip()
        if not q:
            return None, f"step {sid} has an empty question"
        deps = sorted(set(dec) | {int(r) for r in REF.findall(q)})
        bad = [d for d in deps if not 1 <= d < sid]
        if bad:
            return None, f"step {sid} refers to #{bad[0]}, which is not an earlier step"
        if kind in ("CMP", "LOG") and not deps:
            return None, f"step {sid} is {kind} but does not use any earlier result"
        if normalize and kind == "LOG" and len(deps) == 1 and not is_yesno(q):
            kind = "TXT"   # round 4: a non-yes/no question about one earlier result is a look-up, so it gets retrieval
        steps.append({"id": sid, "kind": kind, "question": q, "deps": deps})
    used = {d for s in steps for d in s["deps"]}
    unused = [s["id"] for s in steps[:-1] if s["id"] not in used]
    if unused:
        return None, f"step {unused[0]} is not used by any later step; only the last step may be final"
    if question is not None and is_yesno(steps[-1]["question"]) and not is_yesno(question):
        return None, ("the question asks for a name, date, number or phrase, not for yes or no, so the last step must ask "
                      "for that answer itself instead of a yes/no judgment")
    return steps, None


def fallback_plan(question):
    return [{"id": 1, "kind": "TXT", "question": question, "deps": []}]


def make_plan(llm, question, strict=True):
    """strict=False (e8 ablation) drops the two round-4 rules: the yes/no check on the last step and the LOG->TXT rewrite."""
    msgs = messages(question)
    cost = {"calls": 0, "prompt": 0, "completion": 0}
    first_err = None
    for attempt in (1, 2):
        out = llm.chat(msgs, temperature=0.0, max_tokens=MAX_PLAN_TOKENS)
        cost["calls"] += 1
        cost["prompt"] += out["usage"]["prompt"]
        cost["completion"] += out["usage"]["completion"]
        text = out["texts"][0]
        steps, err = parse(text, question if strict else None, normalize=strict)
        if steps:
            return {"steps": steps, "fallback": False, "attempts": attempt, "error": first_err, "raw": text, "cost": cost}
        first_err = first_err or err
        msgs = msgs + [{"role": "assistant", "content": text},
                       {"role": "user", "content": f"This plan is invalid: {err}. Return a corrected plan as JSON only."}]
    return {"steps": fallback_plan(question), "fallback": True, "attempts": 2, "error": first_err, "raw": text,
            "cost": cost}


# ---- plan structure -------------------------------------------------------------------------------------------------

def children(steps):
    ch = {s["id"]: [] for s in steps}
    for s in steps:
        for d in s["deps"]:
            ch[d].append(s["id"])
    return ch


def structure(steps):
    """Per step id: depth (longest path from a source, sources = 0), number of descendants, in-degree, is_sink."""
    ch = children(steps)
    depth, desc = {}, {}
    for s in steps:
        depth[s["id"]] = 1 + max((depth[d] for d in s["deps"]), default=-1)
    for s in reversed(steps):
        seen = set()
        for c in ch[s["id"]]:
            seen |= {c} | desc[c]
        desc[s["id"]] = seen
    sink = steps[-1]["id"]
    return {s["id"]: {"depth": depth[s["id"]], "n_desc": len(desc[s["id"]]), "indeg": len(s["deps"]),
                      "is_sink": s["id"] == sink, "n_steps": len(steps)} for s in steps}
