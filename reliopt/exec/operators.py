"""Verification operators. Each takes a step context and its default result r0 and returns
{"answer", "cost", "info"}; "cost" counts only the calls the operator itself would issue (verification tokens).

  PROBE          sample 1 of the pool; records agreement with r0, never changes the answer          1 call
  VOTE3 / VOTE5  r0 plus the first 2 / 4 pool samples, majority over normalised answers; NOT FOUND and
                 failed programs abstain; a tie that includes r0 keeps r0                             1 call (n = k-1)
  REGROUND (TXT) rewrite the sub-question as a keyword query (entity name + a few keywords), retrieve REGROUND_K
                 passages without the ones cited so far, re-read; if the two answers differ, the model
                 judges which one its cited passages directly support                                  2-3 calls
  REROUTE (CMP, LOG) re-derive the step by a different route (CMP: a short program that first binds every value to a
                 variable, instead of one expression; LOG: brief written reasoning instead of a direct
                 answer); on disagreement keep the side that passes the type check, and if both do,
                 the model judges which one is right                                                   1-2 calls
  VOTE5+REGROUND / VOTE5+REROUTE  VOTE5 first, then REGROUND / REROUTE checks the vote winner          sum of both
  DEEPGROUND (TXT) deep re-grounding: fuse the BM25 rankings of the sub-question and of its keyword rewrite (reciprocal
                 rank fusion), read the top DEEP_K passages in groups of DEEP_GROUP (each group may answer NOT
                 FOUND), then the model picks, among the current answer and the distinct group answers, the
                 one whose cited passage directly states it                                           2 + DEEP_K/DEEP_GROUP calls

Pool samples are drawn once per (step, inputs) with n = 4 (steps.POOL_N); an operator that reads m samples is charged the
prompt once plus the completion tokens of those m samples, as if it had issued its own request with n = m.
"""
import re
from collections import Counter

from ..eval.metrics import normalize_answer
from . import sandbox
from .steps import (CMP_PROGRAM_SYSTEM, MAX_STEP_TOKENS, NOT_FOUND, ZERO, _cost, add_cost, is_abstain, is_yesno,
                    parse_answer)

OPS = {
    "TXT": ("NONE", "PROBE", "VOTE3", "VOTE5", "REGROUND", "VOTE5+REGROUND", "DEEPGROUND"),
    "CMP": ("NONE", "PROBE", "VOTE3", "VOTE5", "REROUTE", "VOTE5+REROUTE"),
    "LOG": ("NONE", "PROBE", "VOTE3", "VOTE5", "REROUTE", "VOTE5+REROUTE"),
}
# Kind-independent names used by type-level policies: ALT is REGROUND on TXT and REROUTE on CMP / LOG; DEEP is DEEPGROUND
# on TXT and, having no deeper counterpart there, the strongest CMP / LOG operator VOTE5+REROUTE.
GENERIC = ("NONE", "VOTE3", "VOTE5", "ALT", "VOTE5+ALT", "DEEP")


def concrete(op, kind):
    if op == "DEEP":
        return "DEEPGROUND" if kind == "TXT" else "VOTE5+REROUTE"
    alt = "REGROUND" if kind == "TXT" else "REROUTE"
    return op.replace("ALT", alt)


def answer_changing(kind):
    return [o for o in OPS[kind] if o not in ("NONE", "PROBE")]


REGROUND_K = 8
DEEP_K, DEEP_GROUP, RRF_C = 20, 5, 60

REWRITE_SYSTEM = """Write a keyword search query that finds a passage answering the question below: the exact name of the entity the question is about, copied as written, plus one to three keywords for the fact asked (e.g. "director", "born", "founded").
Reply with the query only, on one line."""

ADJ_TXT_SYSTEM = """Two candidate answers to the same sub-question are given, each with the passages it cites.
For each candidate, judge whether its cited passages directly state the answer. Do not use outside knowledge.
Reply in exactly one line: "Supported: A", "Supported: B", or "Supported: NEITHER"."""

REROUTE_SYSTEM = """You complete one step of a multi-step question using the results of earlier steps.
Reason briefly in one to three sentences, working from the earlier results only, then give the answer.
Reply in this format:
Reasoning: <brief reasoning>
Answer: <a short answer of a few words; "yes" or "no" for a yes/no question>"""

GROUP_TXT_SYSTEM = """Answer the sub-question using only the passages below.
The answer is a short phrase copied from the passages: a name, a date, a number or a few words. Never write a sentence.
If these passages do not state the answer, reply "Answer: NOT FOUND" and "Passages: none".
Reply in exactly two lines:
Answer: <short answer or NOT FOUND>
Passages: <the ids of the passages that state the answer, or none>"""

GROUP_TXT_SYSTEM_SPAN = """Answer the sub-question using only the passages below.
Copy the answer word for word from the passage that states it, exactly as the passage writes it: do not shorten, expand or rephrase it. Never write a sentence.
If these passages do not state the answer, reply "Answer: NOT FOUND" and "Passages: none".
Reply in exactly two lines:
Answer: <answer copied from the passage, or NOT FOUND>
Passages: <the ids of the passages that state the answer, or none>"""

ADJ_MULTI_SYSTEM = """Several candidate answers to the same sub-question are given, each with the passages it cites.
Choose the candidate whose cited passages directly state the answer to the sub-question about exactly the entity the sub-question names. Do not use outside knowledge.
Reply in exactly one line: "Supported: <letter>" or "Supported: NONE"."""

ADJ_STEP_SYSTEM = """Two candidate answers to one step of a multi-step question are given, each with how it was derived.
Using only the earlier results, judge which candidate correctly answers the step.
Reply in exactly one line: "Correct: A", "Correct: B", or "Correct: NEITHER"."""


def same(a, b):
    return normalize_answer(a or "") == normalize_answer(b or "")


def vote(cands):
    """Majority over normalised answers; abstentions do not vote; ties go to the earliest candidate (r0 comes first)."""
    valid = [c for c in cands if c.get("answer") and not is_abstain(c["answer"])]
    if not valid:
        return cands[0]
    counts = Counter(normalize_answer(c["answer"]) for c in valid)
    top = max(counts.values())
    winners = {k for k, v in counts.items() if v == top}
    return next(c for c in valid if normalize_answer(c["answer"]) in winners)


def _verdict(text, label):
    m = re.search(rf"{label}\s*:\s*\**\s*(A|B|NEITHER)\b", text, flags=re.I)
    if m:
        return m.group(1).upper()
    m = re.fullmatch(r"\s*\**\s*(A|B|NEITHER)\s*\**\.?\s*", text, flags=re.I)
    return m.group(1).upper() if m else "NEITHER"


class Operators:
    def __init__(self, executor):
        self.ex = executor
        self.llm = executor.llm

    def apply(self, op, ctx, r0):
        kind = ctx["step"]["kind"]
        assert op in OPS[kind], (op, kind)
        if op == "NONE":
            return {"answer": r0["answer"], "cost": dict(ZERO), "info": {}}
        if op == "PROBE":
            p = self.ex.pool(ctx)
            s = p["cands"][0]["answer"]
            return {"answer": r0["answer"], "cost": self._pool_cost(p, 1),
                    "info": {"agree": same(s, r0["answer"]), "probe": s}}
        if op in ("VOTE3", "VOTE5"):
            return self._vote(ctx, r0, int(op[-1]))
        if op == "REGROUND":
            return self._reground(ctx, r0, r0)
        if op == "REROUTE":
            return self._reroute(ctx, r0, r0)
        if op == "DEEPGROUND":
            return self._deepground(ctx, r0, r0)
        v = self._vote(ctx, r0, 5)
        cur = v["info"]["winner"]
        second = self._reground(ctx, r0, cur) if kind == "TXT" else self._reroute(ctx, r0, cur)
        return {"answer": second["answer"], "cost": add_cost(v["cost"], second["cost"]),
                "info": {"vote": v["answer"], **second["info"]}}

    # ---- VOTE / PROBE -------------------------------------------------------------------------------------------------
    @staticmethod
    def _pool_cost(p, m):
        return {"calls": 1, "prompt": p["prompt"], "completion": sum(p["completions"][:m])}

    def _vote(self, ctx, r0, k):
        p = self.ex.pool(ctx)
        cands = [r0] + p["cands"][: k - 1]
        win = vote(cands)
        return {"answer": win["answer"], "cost": self._pool_cost(p, k - 1),
                "info": {"winner": win, "votes": [c.get("answer") for c in cands]}}

    # ---- REGROUND -----------------------------------------------------------------------------------------------------
    def _cited_passages(self, cand, hits):
        ids = [i for i in cand.get("cited", []) if 1 <= i <= len(hits)]
        return [hits[i - 1] for i in ids] if ids else list(hits[:2])

    def _rewrite(self, q):
        rw = self.llm.chat([{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": f"Query: {q}"}],
                           temperature=0.0, max_tokens=64)
        lines = [l.strip() for l in rw["texts"][0].strip().splitlines() if l.strip()]
        q2 = re.sub(r"^(new query|query)\s*:\s*", "", lines[0], flags=re.I).strip().strip('"') if lines else ""
        return q2 or q, _cost(rw)

    def _reground(self, ctx, r0, cur):
        q = ctx["resolved"]
        q2, cost = self._rewrite(q)
        hits0 = r0["hits"]
        cur_pass = self._cited_passages(cur, hits0)
        hits2 = self.ex.retrieve(q2, exclude=[h["id"] for h in cur_pass if cur.get("cited")], k=REGROUND_K)
        out = self.llm.chat(self.ex.txt_messages(q, hits2), temperature=0.0, max_tokens=MAX_STEP_TOKENS, logprobs=True)
        cost = add_cost(cost, _cost(out))
        new = self.ex._txt_cand(out["texts"][0], (out.get("tokens") or [None])[0], (out.get("logprobs") or [None])[0],
                                hits2)
        info = {"query2": q2, "new": new["answer"], "verdict": None}
        if is_abstain(new["answer"]) or same(new["answer"], cur["answer"]):
            final = cur
        elif is_abstain(cur["answer"]):
            final = new
        else:
            user = (f"Sub-question: {q}\n\nCandidate A: {cur['answer']}\nPassages cited by A:\n"
                    + "\n".join(f"- {h['title']}: {h['text']}" for h in cur_pass)
                    + f"\n\nCandidate B: {new['answer']}\nPassages cited by B:\n"
                    + "\n".join(f"- {h['title']}: {h['text']}" for h in self._cited_passages(new, hits2)))
            adj = self.llm.chat([{"role": "system", "content": ADJ_TXT_SYSTEM}, {"role": "user", "content": user}],
                                temperature=0.0, max_tokens=16)
            cost = add_cost(cost, _cost(adj))
            v = _verdict(adj["texts"][0], "Supported")
            info["verdict"] = v
            final = new if v == "B" else cur
        return {"answer": final["answer"], "cost": cost, "info": info}

    # ---- DEEPGROUND ---------------------------------------------------------------------------------------------------
    def _fused(self, queries):
        score, docs = {}, {}
        for q in queries:
            for rank, h in enumerate(self.ex.index.search(q, k=DEEP_K)):
                score[h["id"]] = score.get(h["id"], 0.0) + 1.0 / (RRF_C + rank + 1)
                docs[h["id"]] = h
        return [docs[i] for i in sorted(score, key=lambda i: -score[i])][:DEEP_K]

    def _deepground(self, ctx, r0, cur):
        q = ctx["resolved"]
        q2, cost = self._rewrite(q)
        passages = self._fused([q, q2] if q2 != q else [q])
        options = []   # (answer, cited passages) per distinct candidate; the current answer first
        cur_pass = self._cited_passages(cur, r0["hits"]) if cur.get("cited") else []
        if not is_abstain(cur["answer"]):
            options.append((cur["answer"], cur_pass))
        for i in range(0, len(passages), DEEP_GROUP):
            group = passages[i:i + DEEP_GROUP]
            if self.ex.txt == "span":
                head = [{"role": "system", "content": GROUP_TXT_SYSTEM_SPAN}, *self.ex.TXT_EXAMPLE_FOUND]
            else:   # rounds 1-3, unchanged
                head = [{"role": "system", "content": GROUP_TXT_SYSTEM}, *self.ex.TXT_EXAMPLE_FOUND[:2]]
            out = self.llm.chat(head + [{"role": "user",
                                         "content": f"Passages:\n{self.ex.passages_block(group)}\n\nSub-question: {q}"}],
                                temperature=0.0, max_tokens=64)
            cost = add_cost(cost, _cost(out))
            c = self.ex._txt_cand(out["texts"][0], None, None, group)
            if not is_abstain(c["answer"]) and not any(same(c["answer"], a) for a, _ in options):
                options.append((c["answer"], [group[j - 1] for j in c["cited"]] or group[:1]))
        info = {"query2": q2, "options": [a for a, _ in options], "verdict": None}
        if len(options) <= 1:   # nothing to choose between; adopt a group answer only if the current one abstained
            ans = options[0][0] if options and is_abstain(cur["answer"]) else cur["answer"]
            return {"answer": ans, "cost": cost, "info": info}
        letters = "ABCDEFGH"
        blocks = []
        for (a, ps), L in zip(options, letters):
            cited = "\n".join(f"- {h['title']}: {h['text']}" for h in ps) or "- (no passage; answered from memory)"
            blocks.append(f"Candidate {L}: {a}\nPassages cited by {L}:\n{cited}")
        adj = self.llm.chat([{"role": "system", "content": ADJ_MULTI_SYSTEM},
                             {"role": "user", "content": f"Sub-question: {q}\n\n" + "\n\n".join(blocks)}],
                            temperature=0.0, max_tokens=16)
        cost = add_cost(cost, _cost(adj))
        m = re.search(r"Supported\s*:\s*\**\s*([A-H]|NONE)\b", adj["texts"][0], flags=re.I)
        v = m.group(1).upper() if m else "NONE"
        info["verdict"] = v
        if v != "NONE" and letters.index(v) < len(options):
            return {"answer": options[letters.index(v)][0], "cost": cost, "info": info}
        return {"answer": cur["answer"], "cost": cost, "info": info}

    # ---- REROUTE ------------------------------------------------------------------------------------------------------
    @staticmethod
    def _type_ok(answer, ctx):
        if not answer or is_abstain(answer):
            return False
        yesno = is_yesno(ctx["resolved"]) or (ctx["is_sink"] and is_yesno(ctx["question"]))
        return normalize_answer(answer) in ("yes", "no") if yesno else True

    @staticmethod
    def _derivation(cand):
        if cand.get("exec_ok"):
            return f"computed with the Python expression {sandbox.extract_expression(cand['text'])}"
        return "stated directly"

    def _reroute(self, ctx, r0, cur):
        program = ctx["step"]["kind"] == "CMP"
        out = self.llm.chat([{"role": "system", "content": CMP_PROGRAM_SYSTEM if program else REROUTE_SYSTEM},
                             {"role": "user", "content": self.ex.step_block(ctx)}],
                            temperature=0.0, max_tokens=MAX_STEP_TOKENS, logprobs=True)
        cost = _cost(out)
        new_text = out["texts"][0]
        if program:
            new, err = sandbox.evaluate_program(new_text)
            new = new or NOT_FOUND   # a program that does not run fails the type check below
        else:
            new = parse_answer(new_text) or NOT_FOUND
        info = {"new": new, "verdict": None}
        if same(new, cur["answer"]):
            return {"answer": cur["answer"], "cost": cost, "info": info}
        ok_cur, ok_new = self._type_ok(cur["answer"], ctx), self._type_ok(new, ctx)
        if ok_cur != ok_new:
            info["verdict"] = "type"
            return {"answer": cur["answer"] if ok_cur else new, "cost": cost, "info": info}
        if not ok_cur:
            return {"answer": cur["answer"], "cost": cost, "info": info}
        if program:
            how = "computed with the Python program: " + " ; ".join(l.strip() for l in new_text.strip().splitlines() if l.strip())
        else:
            how = "reasoning: " + re.sub(r"(?is)\banswer\s*:.*$", "", new_text).replace("Reasoning:", "").strip()
        user = (f"{self.ex.step_block(ctx)}\n\nCandidate A: {cur['answer']} ({self._derivation(cur)})\n"
                f"Candidate B: {new} ({how})")
        adj = self.llm.chat([{"role": "system", "content": ADJ_STEP_SYSTEM}, {"role": "user", "content": user}],
                            temperature=0.0, max_tokens=16)
        cost = add_cost(cost, _cost(adj))
        v = _verdict(adj["texts"][0], "Correct")
        info["verdict"] = v
        return {"answer": new if v == "B" else cur["answer"], "cost": cost, "info": info}
