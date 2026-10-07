"""PyRAG, training-free variant (Sun et al., arXiv 2605.12975, 2026), ported from the official code (GasolSun36/PyRAG,
commit 5d8ab2e): pyrag/decompose_agent.py, plan_agent.py, tools.py, runner.py. Prompts are copied verbatim.

Pipeline: decompose the question into atomic sub-queries (JSON list; 3 attempts, else the question itself) -> the plan
agent writes a Python program over retrieve(query) and answer(query, docs) (3 attempts with syntax feedback) -> execute;
a runtime error is sent back for a fix (up to 3 rounds) -> execution-driven adaptive retrieval: if an answer() call returns
an "insufficient information" marker, the program is re-executed with top-k 10 at the retrieve() call feeding it (or at the
last retrieve() when only the final answer is insufficient).
Settings as in scripts/eval.py: one backbone for all roles (the paper's matched-backbone choice here; the authors
recommend a coder model for planning), temperature 0.7, max_tokens 1024, top-k 5, thinking disabled. Retries use seed+i
so that a retry is a fresh sample, as with unseeded sampling in the original.
Differences (disclosed): BM25 over the IRCoT corpora instead of E5 over Wikipedia 2018; the program runs in a restricted
interpreter (no imports, no dunder access, a small set of builtins, at most 40 tool calls) instead of a bare exec().
"""
import ast
import json
import re

TEMPERATURE, MAX_TOKENS, TOPK, BOOST_TOPK = 0.7, 1024, 5, 10
MAX_RETRIES, MAX_FIX_ROUNDS, MAX_TOOL_CALLS = 3, 3, 40

DECOMPOSE_SYSTEM = (
    "You are a question decomposition agent for multi-hop QA. "
    "Break a complex question into a minimal list of atomic single-hop sub-queries. "
    "Each sub-query should be independently answerable by a search engine. "
    "Return a JSON list of strings ONLY. No explanation, no markdown, no extra text."
)
DECOMPOSE_EXAMPLE = """\
Example:
Original question:
In what year was the director of Inception born?

JSON output:
["Who directed Inception?", "When was Christopher Nolan born?"]"""

PLAN_SYSTEM = """You are a planning agent that writes Python code to answer questions via RAG.

You have exactly two functions available:

  retrieve(query: str) -> List[str]
      Searches a retrieval system and returns a list of relevant document strings.
      Do NOT pass a topk argument; the retrieval count is controlled externally.

  answer(query: str, docs: List[str]) -> str
      Calls an LLM to answer a question given a list of documents.
      Returns ONLY the short text inside the model's <answer>...</answer> block (the runtime extracts
      it). Use that return value directly in f-strings — it is a short phrase, not the full model reply.

  answer(query: str) -> str          (no docs — aggregation / synthesis mode)
      Same as above but with NO documents. The model answers using ONLY the information already
      present in the query string. Use this for the FINAL aggregation step.

Rules:
- Use retrieve() + answer() step by step to answer each sub-query.
- If you call retrieve() for a sub-question, pass that return value into the matching answer() for
  that sub-question. Do not pass [] for that step unless the answer is intentionally retrieval-free.
- Build on intermediate answers by interpolating them into later queries using f-strings.
- Variable names MUST be valid Python identifiers: only letters, digits, and underscores, NO spaces.
- Every variable you reference MUST be assigned BEFORE it is used. Never use a name that was not
  explicitly assigned in a previous line.
- Do NOT parse answer() return values in generated code (no .split(), regex, or indexing). The runtime
  already returns the short <answer> span.
- If you use a for/if/while block that might conditionally set final_answer, initialise final_answer = ""
  BEFORE the block so it is always defined.
- The last line of your code MUST be: final_answer = answer(<synthesis question>)
  Do NOT pass docs to this final answer() call — it is aggregation-only.
- NEVER use an f-string or string concatenation as final_answer. final_answer MUST ALWAYS be the
  return value of an answer() call so the result goes through the model and has a proper <answer> tag.
- CRITICAL: The <synthesis question> must use this TWO-PART format:
    "Given: <fact1>, <fact2>, ... Answer the question: <ORIGINAL QUESTION>"
  where the first part lists intermediate results as background facts, and the second part REPEATS
  the ORIGINAL question VERBATIM after "Answer the question:".
  Do NOT rephrase the original question or embed intermediate answers INTO the question itself.
  BAD:  f"Which American director, {name}, who is {nationality}, hosted ...?"  (answer leaks into the question → LLM says "yes")
  GOOD: f"Given: {name} hosted the event, {name} is {nationality}. Answer the question: Which American director hosted ...?"
- Valid Python 3 only: use ASCII double or single quotes for strings; never put backticks around
  variable names or expressions.
- Return ONLY the Python code, no explanation, no imports, no markdown prose."""

CODE_EXAMPLE = """\
# Example for: "Who was born earlier, the director of Inception or Jurassic Park?"
docs1 = retrieve("Who directed Inception?")
director1 = answer("Who directed Inception?", docs1)

docs2 = retrieve("Who directed Jurassic Park?")
director2 = answer("Who directed Jurassic Park?", docs2)

docs3 = retrieve(f"When was {director1} born?")
birth1 = answer(f"When was {director1} born?", docs3)

docs4 = retrieve(f"When was {director2} born?")
birth2 = answer(f"When was {director2} born?", docs4)

# ALWAYS end with answer() — list facts first, then repeat the ORIGINAL question verbatim
final_answer = answer(
    f"Given: {director1} directed Inception and was born {birth1}, "
    f"{director2} directed Jurassic Park and was born {birth2}. "
    f"Answer the question: Who was born earlier, the director of Inception or Jurassic Park?"
)"""

FIX_PROMPT_TEMPLATE = """\
The following Python code was generated to answer the question but raised a runtime error.
Fix the code so it runs correctly. Return ONLY the corrected Python code.

=== Original question ===
{original_query}

=== Failing code ===
{failed_code}

=== Runtime error ===
{error_msg}

=== Fix instructions ===
- Do not parse answer() return values in generated code; the runtime returns short <answer> text.
- Every variable you reference must be explicitly assigned earlier.
- If retrieve() was called, pass its docs into answer(); do not use answer(..., []) for that step.
- Initialise final_answer = "" before any loop/conditional that might set it.
- The last line MUST be: final_answer = answer(<synthesis question>) — call answer() with NO docs
  to aggregate all intermediate results. NEVER use an f-string or string as final_answer directly.
"""

SYNTAX_FEEDBACK_TEMPLATE = """
---
Your previous output is NOT valid Python (syntax error).
Details: {error_detail}

Fix it. Use only ASCII ' or \" for strings; never wrap variable names in backticks; output code only.

Previous attempt:
```python
{failed_code}
```
Return ONLY corrected Python code.
"""

ANSWER_SYSTEM_WITH_DOCS = (
    "You are given a question and retrieved documents.\n"
    "You MUST answer using ONLY information from the retrieved documents.\n"
    "Even for yes/no questions, decide yes or no by reasoning from facts in the documents.\n\n"
    "Output format (STRICT):\n"
    "<redacted_thinking> ... </redacted_thinking>\n"
    "<answer> ... </answer>\n\n"
    "Evidence citation rule:\n"
    "- Whenever you use evidence from the documents in your reasoning, you MUST cite it inline as Doc [i] "
    "(matching the document indices shown in the retrieved block, e.g. [Doc 1] → Doc [1]).\n"
    "- Only cite documents that are actually relevant.\n"
    "- Keep <redacted_thinking> concise (1–3 sentences).\n\n"
    "Answer rules:\n"
    "- The <answer> should be a short phrase, preferably taken directly from the documents when possible.\n"
    "- Match the answer TYPE to the QUESTION: WHO / which person / 谁先 / born first / earlier → a person's "
    "NAME in <answer>, not only a date; WHEN / 何时 → date or time; yes/no → exactly yes / no / unknown when "
    "the documents do not support a definite answer.\n"
    "- Do NOT output anything outside <redacted_thinking> and <answer>.\n\n"
    "Example (do NOT copy the content, only follow the style):\n"
    "<redacted_thinking>Doc [1] states that Future Ted serves as the narrator, and Doc [4] confirms the voice actor.</redacted_thinking>\n"
    "<answer> Ted Mosby </answer>\n"
)
ANSWER_SYSTEM_NO_DOCS = (
    "There are NO retrieved documents. The question text itself contains background facts "
    "(after 'Given:') and the actual question to answer (after 'Answer the question:').\n"
    "You MUST use the provided facts to answer the ACTUAL QUESTION.\n\n"
    "CRITICAL: Your job is to ANSWER the question, NOT to confirm whether the facts are true.\n"
    "- If the question asks WHO / WHICH person → reply with a person's NAME.\n"
    "- If the question asks WHEN → reply with a date or time.\n"
    "- If the question asks WHERE → reply with a place.\n"
    "- ONLY answer yes/no when the question is explicitly a yes/no question (e.g. 'Are both ...?', 'Is it true ...?').\n"
    "- NEVER answer 'yes' or 'no' to a WHO/WHICH/WHEN/WHERE question.\n\n"
    "Output format (STRICT):\n"
    "<redacted_thinking> ... </redacted_thinking>\n"
    "<answer> ... </answer>\n\n"
    "- In <redacted_thinking>, identify the actual question type and combine the given facts to produce the answer (1–2 sentences).\n"
    "- The <answer> must directly answer the question — a name, date, place, etc. — NOT 'yes' or 'no' unless the question is truly yes/no.\n"
    "- Do NOT output anything outside <redacted_thinking> and <answer>.\n"
)
INSUFFICIENT = ("not enough information", "insufficient information", "no enough information", "cannot answer",
                "unable to answer", "信息不足", "无法根据", "无法回答", "unknown", "not found")
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
                  for n in ("len", "str", "int", "float", "bool", "range", "list", "dict", "set", "tuple", "min", "max",
                            "sorted", "enumerate", "zip", "abs", "round", "isinstance", "any", "all", "sum", "reversed")}
_SAFE_BUILTINS["print"] = lambda *a, **k: None


def extract_json_block(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```json\s*(.*?)\s*```", text, re.S | re.I)
    if m:
        return json.loads(m.group(1).strip())
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if m:
        return json.loads(m.group(1).strip())
    raise ValueError("could not parse JSON")


def extract_python_block(text):
    text = text.strip()
    m = re.search(r"```python\s*(.*?)\s*```", text, re.S | re.I) or re.search(r"```\s*(.*?)\s*```", text, re.S)
    code = m.group(1).strip() if m else text
    code = code.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", " ": " "}))
    return re.sub(r"[​-‍﻿]", "", code)


def extract_answer_tag(text):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.S | re.I)
    return m.group(1).strip() if m else text.strip()


def insufficient(text):
    return isinstance(text, str) and bool(text) and any(m in text.strip().lower() for m in INSUFFICIENT)


def _check_ast(code):
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Lambda, ast.ClassDef,
                             ast.AsyncFunctionDef, ast.Await)):
            raise RuntimeError(f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise RuntimeError("access to private attributes is not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise RuntimeError("dunder names are not allowed")


class PyRAG:
    def __init__(self, llm, index, seed=17):
        self.llm, self.index, self.seed = llm, index, seed

    def _gen(self, system, user, attempt=0):
        out = self.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                            temperature=TEMPERATURE, max_tokens=MAX_TOKENS, seed=self.seed + attempt)
        self.tok += out["usage"]["prompt"] + out["usage"]["completion"]; self.calls += 1
        return _THINK.sub("", out["texts"][0]).strip()

    def decompose(self, query):
        user = (f"Original question:\n{query}\n\n{DECOMPOSE_EXAMPLE}\n\n"
                "Now return a JSON list of atomic sub-queries for the original question above.\n"
                "Return ONLY the JSON list of strings.")
        for a in range(MAX_RETRIES):
            try:
                res = extract_json_block(self._gen(DECOMPOSE_SYSTEM, user, a))
                if isinstance(res, list) and res and all(isinstance(x, str) for x in res):
                    return res
            except Exception:
                pass
        return [query]

    def generate_code(self, query, subs):
        user = (f"Original question:\n{query}\n\nSub-queries to resolve:\n{subs}\n\n"
                f"Reference example (do NOT copy, write code for the actual question above):\n{CODE_EXAMPLE}\n\n"
                "Now write the Python code for the original question.\n"
                f'End with: final_answer = answer(f"Given: <facts>. Answer the question: {query}")')
        extra = ""
        for a in range(MAX_RETRIES):
            raw = self._gen(PLAN_SYSTEM, user + extra, a)
            code = extract_python_block(raw)
            try:
                compile(code, "<generated>", "exec")
                return code
            except SyntaxError as e:
                detail = f"{e.msg} (line {e.lineno})" if e.lineno else f"{e.msg} (unknown)"
                extra = SYNTAX_FEEDBACK_TEMPLATE.format(error_detail=detail, failed_code=code[:6000] or "(empty)")
        return None

    def fix_code(self, query, code, err):
        user = FIX_PROMPT_TEMPLATE.format(original_query=query, failed_code=code, error_msg=err)
        for a in range(MAX_RETRIES):
            fixed = extract_python_block(self._gen(PLAN_SYSTEM, user, a))
            try:
                compile(fixed, "<generated>", "exec")
                return fixed
            except SyntaxError:
                continue
        return None

    def _execute(self, code, boost):
        log, idx = [], [0]

        def retrieve(query, topk=TOPK):
            if len(log) >= MAX_TOOL_CALLS:
                raise RuntimeError("too many tool calls")
            idx[0] += 1
            topk = max(topk, boost.get(idx[0], topk))
            hits = self.index.search(str(query), k=topk)
            docs = [f"Doc {i + 1} (Title: {h['title']})\n{h['text']}" for i, h in enumerate(hits)]
            log.append({"type": "retrieve", "topk": topk})
            return docs

        def answer(query, docs=None):
            if len(log) >= MAX_TOOL_CALLS:
                raise RuntimeError("too many tool calls")
            docs = list(docs or [])
            block = "\n\n".join(f"[Doc {i + 1}]\n{d}" for i, d in enumerate(docs)) if docs else "No retrieved documents."
            user = (f"=== QUESTION ===\n{query}\n=== END QUESTION ===\n\n"
                    f"=== RETRIEVED DOCUMENTS ===\n{block}\n=== END DOCUMENTS ===")
            ret = extract_answer_tag(self._gen(ANSWER_SYSTEM_WITH_DOCS if docs else ANSWER_SYSTEM_NO_DOCS, user))
            log.append({"type": "answer", "answer": ret})
            return ret

        ns = {"__builtins__": _SAFE_BUILTINS, "retrieve": retrieve, "answer": answer}
        try:
            _check_ast(code)
            exec(compile(code, "<generated>", "exec"), ns)
        except Exception as e:
            raise RuntimeError(f"Code execution failed ({type(e).__name__}: {e})\n--- Generated code ---\n{code}")
        fa = ns.get("final_answer", "[ERROR: final_answer not set]")
        return {"final_answer": "" if callable(fa) else str(fa), "log": log}

    def _execute_with_fixes(self, query, code, boost):
        for r in range(MAX_FIX_ROUNDS + 1):
            try:
                return self._execute(code, boost), code
            except RuntimeError as e:
                if r == MAX_FIX_ROUNDS:
                    return None, code
                fixed = self.fix_code(query, code, str(e))
                if fixed is None:
                    return None, code
                code = fixed
        return None, code

    @staticmethod
    def _boost(log, final_answer):
        out, n_ret = {}, 0
        topks = {}
        for e in log:
            if e["type"] == "retrieve":
                n_ret += 1
                topks[n_ret] = e["topk"]
            elif insufficient(e["answer"]) and n_ret > 0 and topks[n_ret] < BOOST_TOPK:
                out[n_ret] = BOOST_TOPK
        if not out and insufficient(final_answer) and n_ret > 0 and topks[n_ret] < BOOST_TOPK:
            out[n_ret] = BOOST_TOPK
        return out

    def run(self, q):
        self.tok = self.calls = 0
        query = q["question"]
        subs = self.decompose(query)
        code = self.generate_code(query, subs)
        status = "ok"
        result = None
        if code is None:
            status = "plan_failed"
        else:
            result, code = self._execute_with_fixes(query, code, {})
            if result is None:
                status = "exec_failed"
            else:
                boost = self._boost(result["log"], result["final_answer"])
                if boost:
                    r2, code = self._execute_with_fixes(query, code, boost)
                    status = "boosted"
                    if r2 is not None:
                        result = r2
        pred = result["final_answer"] if result else ""
        return {"pred": pred, "tok": self.tok, "calls": self.calls, "status": status, "subs": subs,
                "code": (code or "")[:2000]}
