"""Single-pass reference methods: CoT, Standard RAG and Self-Consistency over Standard RAG.

CoT: zero-shot chain of thought without retrieval [R23]. Standard RAG: retrieve the top-k BM25 passages for the original
question once, reason briefly, answer. Self-Consistency [R08]: SC_POOL samples of the Standard RAG prompt at temperature
0.7 / top-p 0.95 in one request with a fixed seed; SC-k votes over the first k (normalised answers, ties to the earliest).
Every method ends with "Answer: <short answer>" and is scored with the same normalisation; cost is prompt + completion
tokens, with SC-k charged the prompt once plus the completions of the k samples it reads.
"""
from collections import Counter
from math import comb

from ..eval.metrics import normalize_answer, score
from ..exec.steps import is_abstain, parse_answer

MAX_TOKENS = 256
SC_POOL = 10
AC_CONF = 0.95
SAMPLE_T, SAMPLE_P = 0.7, 0.95

FORMAT = ("Think step by step in a few sentences, then give the final answer on the last line as \"Answer: <short answer>\". "
          "The short answer is a name, a date, a number, a few words, or yes/no; never a sentence.")
COT_SYSTEM = "Answer the question. " + FORMAT
RAG_SYSTEM = "Answer the question using the passages below; if they are not enough, also use your own knowledge. " + FORMAT


def cot_messages(question):
    return [{"role": "system", "content": COT_SYSTEM}, {"role": "user", "content": f"Question: {question}"}]


def rag_messages(question, hits):
    ctx = "\n".join(f"[{i}] {h['title']}: {h['text']}" for i, h in enumerate(hits, 1)) or "(no passages found)"
    return [{"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: {question}"}]


def _row(pred, golds, tok):
    return {"pred": pred, **score(pred, golds), "tok": tok}


def majority(answers):
    valid = [a for a in answers if a and not is_abstain(a)]
    if not valid:
        return answers[0] if answers else ""
    counts = Counter(normalize_answer(a) for a in valid)
    top = max(counts.values())
    return next(a for a in valid if counts[normalize_answer(a)] == top)


def beta_stop_prob(a, b):
    """P(p > 1/2) for p ~ Beta(a + 1, b + 1): the quantity BetaStoppingCriteria integrates numerically (Adaptive-Consistency,
    adaptive_consistency/stopping_criterias.py). For integer counts it equals P(Binomial(a + b + 1, 1/2) <= a)."""
    n = a + b + 1
    return sum(comb(n, j) for j in range(a + 1)) / 2 ** n


def adaptive_consistency(answers, completions, prompt_tokens, conf=AC_CONF):
    """Adaptive-Consistency [R10] replayed over a Self-Consistency pool (official code, Pranjal2041/AdaptiveConsistency
    commit 899ab56: AC.eval_loop with step_size 1 and BetaStoppingCriteria(0.95)): read the samples in order, after each one
    count the two most frequent answers (a, b) and stop once P(p > 1/2 | Beta(a + 1, b + 1)) >= 0.95, or when the pool
    (= max_gens, 10 here instead of the paper's 40) runs out. Counting uses the same normalisation and abstention filter
    as SC voting, and the answer is the SC majority of the samples read. Cost is charged as for SC-k (prompt once plus
    the completions read), although the official loop issues one request per sample; this favours AC.
    Returns (answer, number of samples read, tokens)."""
    for m in range(1, len(answers) + 1):
        votes = Counter(normalize_answer(x) for x in answers[:m] if x and not is_abstain(x)).most_common(2)
        a = votes[0][1] if votes else 0
        b = votes[1][1] if len(votes) > 1 else 0
        if a and beta_stop_prob(a, b) >= conf:
            break
    return majority(answers[:m]), m, prompt_tokens + sum(completions[:m])


def run_simple(llm, index, q, k=5, seed=17, sc_ks=(3, 5, 10)):
    """All first-batch methods on one question: {"cot", "rag", "sc<k>"...: {pred, em, f1, acc, tok}, "sc_answers": [...]}."""
    golds = q["answers"]
    out = {}
    c = llm.chat(cot_messages(q["question"]), temperature=0.0, max_tokens=MAX_TOKENS)
    out["cot"] = _row(parse_answer(c["texts"][0]), golds, c["usage"]["prompt"] + c["usage"]["completion"])
    msgs = rag_messages(q["question"], index.search(q["question"], k=k))
    r = llm.chat(msgs, temperature=0.0, max_tokens=MAX_TOKENS)
    out["rag"] = _row(parse_answer(r["texts"][0]), golds, r["usage"]["prompt"] + r["usage"]["completion"])
    s = llm.chat(msgs, temperature=SAMPLE_T, top_p=SAMPLE_P, max_tokens=MAX_TOKENS, n=SC_POOL, seed=seed, logprobs=True)
    answers = [parse_answer(t) for t in s["texts"]]
    comps = [len(l) for l in s["logprobs"]] if s.get("logprobs") else [s["usage"]["completion"] // SC_POOL] * SC_POOL
    for kk in sc_ks:
        out[f"sc{kk}"] = _row(majority(answers[:kk]), golds, s["usage"]["prompt"] + sum(comps[:kk]))
    out["sc_answers"] = answers
    out["sc_completions"] = comps
    return out
