"""Step execution for TXT / CMP / LOG steps and the shared pool of sampled re-executions (§4.3).

A step context `ctx` holds the original question, the step, its sub-question with "#k" replaced by the actual earlier
results, and the earlier results it uses. Everything a step sees is in its prompt, so the LLM cache key contains the actual
parent answers: a policy that changes a parent answer re-executes the children, and only new input combinations call the
model.

Default execution is greedy with token logprobs. TXT retrieves the top-k BM25 passages for the resolved sub-question and
reads them; it always answers (from its own knowledge with "Passages: none" when the passages lack the answer, and if the
reader still abstains, one closed-book call answers instead), because an abstention would leave every later step without an
input; CMP asks for one Python expression and runs it in the sandbox (if that fails, the step is answered directly with
the LOG prompt, and the fallback call counts as execution); LOG combines earlier results.
The sample pool is one request with n = POOL_N samples at temperature 0.7, top-p 0.95 and a fixed seed; PROBE and VOTE-k
read its first samples, so every operator on a step reuses the same samples (and the cost of each operator counts only the
samples it reads, see operators.py).
"""
import re

from ..eval.metrics import normalize_answer
from . import sandbox

K_PASSAGES = 5
K_SINK_Q = 5   # passages retrieved for the original question at a TXT sink (sink_q, round 4)
MAX_STEP_TOKENS = 256
SAMPLE_T, SAMPLE_P = 0.7, 0.95
POOL_N = 4
NOT_FOUND = "NOT FOUND"
REF = re.compile(r"#(\d+)")
_YESNO_START = re.compile(r"^(is|are|was|were|do|does|did|has|have|had|can|could|will|would|should)\b", re.I)

TXT_SYSTEM = """Answer the sub-question using the passages below.
The answer is a short phrase, copied from the passages when possible: a name, a date, a number or a few words. Never write a sentence and do not add descriptions.
Always give an answer. If the passages do not contain it, give your best guess from your own knowledge and write "Passages: none"; never answer "none" or "not found".
Reply in exactly two lines:
Answer: <short answer>
Passages: <the ids of the passages that state the answer, e.g. 2, 4, or none>"""

# Two examples, so the reply format is shown rather than only described: a fictional look-up answered from a passage, and a
# well-known fact the passages miss, answered from the model's own knowledge (an abstention would starve every later step).
TXT_EXAMPLE = [
    {"role": "user", "content": "Passages:\n[1] Northern Shore: Northern Shore is a 1987 drama film directed by Lena Varga "
                                "and produced by Orbit Pictures.\n[2] Lena Varga: Lena Varga (born 3 May 1941 in Szeged) is "
                                "a Hungarian film director and screenwriter.\n\nSub-question: Who directed the film Northern Shore?"},
    {"role": "assistant", "content": "Answer: Lena Varga\nPassages: 1"},
    {"role": "user", "content": "Passages:\n[1] Mozart family: The Mozart family moved to Vienna in 1781.\n[2] Salzburg "
                                "Festival: The Salzburg Festival is a music and drama festival held each summer.\n\n"
                                "Sub-question: In which city was Wolfgang Amadeus Mozart born?"},
    {"role": "assistant", "content": "Answer: Salzburg\nPassages: none"},
]

# "span" style (round 4): the answer is the passage's own words. MuSiQue's gold answers are the spans of its single-hop
# source questions ("Pardis County", "Hon July Moyo"), and a reader told to write a short phrase trims them; the added
# example shows a span that keeps its qualifier.
TXT_SYSTEM_SPAN = """Answer the sub-question using the passages below.
Copy the answer word for word from the passage that states it, exactly as the passage writes it: do not shorten, expand or rephrase it. The answer is a name, a date, a number or a few words, never a sentence.
Always give an answer. If the passages do not contain it, give your best guess from your own knowledge and write "Passages: none"; never answer "none" or "not found".
Reply in exactly two lines:
Answer: <answer copied from the passage>
Passages: <the ids of the passages that state the answer, e.g. 2, 4, or none>"""

TXT_EXAMPLE_SPAN = TXT_EXAMPLE[:2] + [
    {"role": "user", "content": "Passages:\n[1] Kestrel: Kestrel is the largest town of Kestrel County.\n[2] Halvard Bay: "
                                "Halvard Bay is a small bay on the coast of Kestrel County in northern Norland.\n\n"
                                "Sub-question: In which county is Halvard Bay?"},
    {"role": "assistant", "content": "Answer: Kestrel County\nPassages: 2"},
] + TXT_EXAMPLE[2:]
TXT_STYLES = {"short": (TXT_SYSTEM, TXT_EXAMPLE), "span": (TXT_SYSTEM_SPAN, TXT_EXAMPLE_SPAN)}

CMP_SYSTEM = """You compute one step of a multi-step question from the results of earlier steps.
Write a single Python expression that evaluates to the answer of this step. Copy the values you need from the earlier results.
You may use numbers, strings, comparisons, arithmetic, `x if condition else y`, and date(year, month, day). When only years are known, compare the years.
Examples:
Which was founded earlier, Arlen Motors (founded 12 May 1921) or Bexley Foods (founded 1930)?
Expression: "Arlen Motors" if 1921 < 1930 else "Bexley Foods"
How many years passed between 3 March 1850 and 1 June 1902?
Expression: 1902 - 1850
Reply in exactly one line:
Expression: <python expression>"""

CLOSED_BOOK_SYSTEM = """Answer the question from your own knowledge with a short phrase: a name, a date, a number or a few words, never a sentence. Always give your best guess.
Reply in exactly one line:
Answer: <short answer>"""

CMP_PROGRAM_SYSTEM = """You compute one step of a multi-step question from the results of earlier steps.
Write a short Python program: first assign every value you need from the earlier results to a variable (dates as date(year, month, day), or just the year when only the year is known; numbers as numbers; names as strings), then on the last line write one expression that gives the answer of this step.
You may use numbers, strings, comparisons, arithmetic, `x if condition else y`, and date(year, month, day). Reply with the program only."""

LOG_SYSTEM = """You complete one step of a multi-step question using the results of earlier steps.
Combine the results, judge yes/no, or state the final answer, as the step asks. Use only the results given.
Always give a short answer, never a sentence; if the results are not enough, give your best guess.
Reply in exactly one line:
Answer: <a short answer of a few words; "yes" or "no" for a yes/no question>"""


def resolve(text, answers):
    """Replace #k by the answer of step k."""
    return REF.sub(lambda m: str(answers.get(int(m.group(1)), m.group(0))), text)


def is_yesno(text):
    return bool(_YESNO_START.match(text.strip()))


def clean_answer(a):
    a = a.strip().strip('"').strip("'").strip()
    a = re.sub(r"\s+", " ", a)
    if a.endswith(".") and not re.search(r"\b[A-Z]\.$", a):
        a = a[:-1].strip()
    low = a.lower()
    for yn in ("yes", "no"):
        if low == yn or re.match(rf"^{yn}\b[,.;:]", low):
            return yn
    return a


def is_abstain(a):
    n = normalize_answer(a or "")
    return not n or n in ("not found", "none", "unknown")


def parse_answer(text):
    """Value after the last "Answer:" (up to the end of that line); the first line when there is no such label."""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    for l in reversed(lines):
        m = re.match(r"^\s*\**\s*answer\s*\**\s*:\s*(.*)$", l, flags=re.I)
        if m:
            a = clean_answer(m.group(1))
            if not re.match(r"^passages?\s*:", a, flags=re.I):
                return a
            # "Answer: Passages: none" leaves the answer empty; the model sometimes puts it on a "Guess:" line instead
            g = [re.sub(r"^\s*\**\s*guess\s*\**\s*:\s*", "", x, flags=re.I) for x in lines if re.match(r"^\s*\**\s*guess\s*\**\s*:", x, flags=re.I)]
            return clean_answer(g[0]) if g else ""
    first = clean_answer(lines[0]) if lines else ""
    return "" if re.match(r"^passages?\s*:", first, flags=re.I) else first


def parse_cited(text, n_passages):
    for l in text.splitlines():
        m = re.match(r"^\s*\**\s*passages?\s*\**\s*:\s*(.*)$", l, flags=re.I)
        if m:
            ids = sorted({int(x) for x in re.findall(r"\d+", m.group(1)) if 1 <= int(x) <= n_passages})
            return ids
    return []


def span_logprobs(tokens, logprobs, label):
    """Mean and min logprob of the tokens that make up the value after the last `label:` in the generated text."""
    if not tokens or not logprobs:
        return None, None
    text, starts = "", []
    for t in tokens:
        starts.append(len(text))
        text += t
    i = text.lower().rfind(label.lower() + ":")
    if i < 0:
        lo, hi = 0, len(text)
    else:
        lo = i + len(label) + 1
        nl = text.find("\n", lo)
        hi = nl if nl >= 0 else len(text)
    sel = [lp for s, t, lp in zip(starts, tokens, logprobs) if s < hi and s + len(t) > lo and t.strip()]
    if not sel:
        return None, None
    return sum(sel) / len(sel), min(sel)


def _cost(out, prompt=True, completion=None):
    return {"calls": 1, "prompt": out["usage"]["prompt"] if prompt else 0,
            "completion": out["usage"]["completion"] if completion is None else completion}


def add_cost(a, b):
    return {k: a.get(k, 0) + b.get(k, 0) for k in ("calls", "prompt", "completion")}


ZERO = {"calls": 0, "prompt": 0, "completion": 0}


def tokens_of(cost):
    return cost["prompt"] + cost["completion"]


class StepExecutor:
    """`txt` picks the TXT reader prompt ("short" up to round 3, "span" from round 4); `k` is the TXT retrieval depth."""

    def __init__(self, llm, index, k=K_PASSAGES, seed=17, txt="short", sink_q=False, cmp_direct=False, untyped=False):
        """sink_q (round 4): a TXT sink also reads the top-K_SINK_Q passages for the original question and sees the
        question itself, so the last look-up can recover the context its sub-question lost.
        Execution-layer ablations (e8; the defaults keep the method's behaviour): cmp_direct answers a CMP step with the
        LOG prompt instead of a program; untyped executes every step as a TXT look-up (retrieve and read), whatever its type."""
        self.llm, self.index, self.k, self.seed, self.txt, self.sink_q = llm, index, k, seed, txt, sink_q
        self.cmp_direct, self.untyped = cmp_direct, untyped
        self.txt_system, self.txt_example = TXT_STYLES[txt]
        # the examples answered from a passage (DEEPGROUND's group reads may say NOT FOUND, so not the closed-book one)
        self.TXT_EXAMPLE_FOUND = [m for m in self.txt_example if m not in TXT_EXAMPLE[2:]]

    # ---- prompts --------------------------------------------------------------------------------------------------
    @staticmethod
    def inputs_block(ctx):
        lines = [f"#{d} ({q}): {a}" for d, q, a in ctx["inputs"]]
        return "Earlier results:\n" + ("\n".join(lines) if lines else "(none)")

    def step_block(self, ctx):
        s = f"{self.inputs_block(ctx)}\n\nStep: {ctx['resolved']}"
        if ctx["is_sink"]:
            s += f"\n(This is the last step. Original question: {ctx['question']})"
        return s

    def retrieve(self, query, exclude=(), k=None):
        k = k or self.k
        hits = self.index.search(query, k=k + len(exclude))
        ex = set(exclude)
        return [h for h in hits if h["id"] not in ex][:k]

    @staticmethod
    def passages_block(hits):
        return "\n".join(f"[{i}] {h['title']}: {h['text']}" for i, h in enumerate(hits, 1)) or "(no passages found)"

    def txt_messages(self, question, hits, original=None):
        user = f"Passages:\n{self.passages_block(hits)}\n\nSub-question: {question}"
        if original:
            user += f"\n(This is the last step of answering the question: {original})"
        return [{"role": "system", "content": self.txt_system}, *self.txt_example, {"role": "user", "content": user}]

    def cmp_messages(self, ctx):
        return [{"role": "system", "content": CMP_SYSTEM}, {"role": "user", "content": self.step_block(ctx)}]

    def log_messages(self, ctx):
        return [{"role": "system", "content": LOG_SYSTEM}, {"role": "user", "content": self.step_block(ctx)}]

    # ---- parsing of one generated sample --------------------------------------------------------------------------
    def _txt_cand(self, text, toks, lps, hits):
        ans = parse_answer(text)
        cited = parse_cited(text, len(hits))
        m, mn = span_logprobs(toks, lps, "Answer")
        return {"answer": NOT_FOUND if is_abstain(ans) else ans, "cited": cited, "lp_mean": m, "lp_min": mn,
                "text": text}

    def _cmp_cand(self, text, toks, lps):
        ans, err = sandbox.evaluate(text)
        m, mn = span_logprobs(toks, lps, "Expression")
        return {"answer": ans, "exec_ok": err is None, "exec_error": err, "lp_mean": m, "lp_min": mn, "text": text}

    def _log_cand(self, text, toks, lps):
        ans = parse_answer(text)
        m, mn = span_logprobs(toks, lps, "Answer")
        return {"answer": ans, "lp_mean": m, "lp_min": mn, "text": text}

    def _parse(self, kind, out, i, hits):
        text = out["texts"][i]
        toks = out["tokens"][i] if out.get("tokens") else None
        lps = out["logprobs"][i] if out.get("logprobs") else None
        if kind == "TXT":
            return self._txt_cand(text, toks, lps, hits)
        if kind == "CMP":
            return self._cmp_cand(text, toks, lps)
        return self._log_cand(text, toks, lps)

    def exec_kind(self, kind):
        """The executor a step of type `kind` runs on (its own type unless an e8 ablation switch says otherwise)."""
        if self.untyped:
            return "TXT"
        return "LOG" if kind == "CMP" and self.cmp_direct else kind

    def _messages(self, ctx):
        kind = self.exec_kind(ctx["step"]["kind"])
        if kind == "TXT":
            hits = self.retrieve(ctx["resolved"])
            if self.sink_q and ctx["is_sink"] and ctx["resolved"] != ctx["question"]:
                seen = {h["id"] for h in hits}
                hits = hits + [h for h in self.index.search(ctx["question"], k=K_SINK_Q) if h["id"] not in seen]
                return self.txt_messages(ctx["resolved"], hits, original=ctx["question"]), hits
            return self.txt_messages(ctx["resolved"], hits), hits
        if kind == "CMP":
            return self.cmp_messages(ctx), None
        return self.log_messages(ctx), None

    # ---- default execution and the sample pool --------------------------------------------------------------------
    def default(self, ctx):
        """Greedy execution of one step. Returns a result dict with the answer, signals for features, and the cost."""
        kind = self.exec_kind(ctx["step"]["kind"])
        msgs, hits = self._messages(ctx)
        out = self.llm.chat(msgs, temperature=0.0, max_tokens=MAX_STEP_TOKENS, logprobs=True)
        r = self._parse(kind, out, 0, hits)
        r.update(kind=kind, cost=_cost(out), truncated=out.get("finish", [None])[0] == "length", fallback=False)
        if kind == "TXT":
            r["hits"] = hits
        if kind == "TXT" and is_abstain(r["answer"]):   # the reader abstained: answer closed-book, counted as execution
            fb = self.llm.chat([{"role": "system", "content": CLOSED_BOOK_SYSTEM},
                                {"role": "user", "content": f"Question: {ctx['resolved']}"}],
                               temperature=0.0, max_tokens=MAX_STEP_TOKENS, logprobs=True)
            c = self._log_cand(fb["texts"][0], (fb.get("tokens") or [None])[0], (fb.get("logprobs") or [None])[0])
            if not is_abstain(c["answer"]):
                r.update(answer=c["answer"], lp_mean=c["lp_mean"], lp_min=c["lp_min"], cited=[], fallback=True,
                         fallback_text=c["text"])
            r["cost"] = add_cost(r["cost"], _cost(fb))
        if kind == "CMP" and not r["exec_ok"]:
            fb = self.llm.chat(self.log_messages(ctx), temperature=0.0, max_tokens=MAX_STEP_TOKENS, logprobs=True)
            c = self._log_cand(fb["texts"][0], (fb.get("tokens") or [None])[0], (fb.get("logprobs") or [None])[0])
            r.update(answer=c["answer"], lp_mean=c["lp_mean"], lp_min=c["lp_min"], fallback=True,
                     fallback_text=c["text"], cost=add_cost(r["cost"], _cost(fb)))
        if not r.get("answer"):
            r["answer"] = NOT_FOUND
        return r

    def pool(self, ctx):
        """POOL_N samples of the default prompt: {"cands": [...], "prompt": P, "completions": [C_1..C_n]}."""
        kind = self.exec_kind(ctx["step"]["kind"])
        msgs, hits = self._messages(ctx)
        out = self.llm.chat(msgs, temperature=SAMPLE_T, top_p=SAMPLE_P, max_tokens=MAX_STEP_TOKENS, n=POOL_N,
                            seed=self.seed, logprobs=True)
        cands = [self._parse(kind, out, i, hits) for i in range(len(out["texts"]))]
        comps = [len(l) for l in out["logprobs"]] if out.get("logprobs") else [out["usage"]["completion"] // POOL_N] * POOL_N
        return {"cands": cands, "prompt": out["usage"]["prompt"], "completions": comps, "hits": hits}
