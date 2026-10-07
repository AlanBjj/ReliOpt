"""IRCoT (Trivedi et al., ACL 2023), reimplemented after the official code (StonyBrookNLP/ircot, commit 3c1820f).

Mirrors base_configs/ircot_qa_codex_<ds>.jsonnet and commaqa/inference/ircot.py:
  - retrieval: K paragraphs for the question with wh-words removed; afterwards K paragraphs for the last generated
    non-reasoning sentence; paragraphs are accumulated (no duplicates) up to 15;
  - reasoning: one CoT sentence per step, few-shot prompt = demonstrations of
    prompts/<ds>/gold_with_2_distractors_context_cot_qa_codex.txt (the 20 annotated qids, in the config's order, dropped from
    the end until the estimated length fits 8000 tokens), then "<paragraphs>\\n\\nQ: <question>\\nA: <cot so far>";
    stop when a sentence matches ".* answer is" or after 10 sentences;
  - reading: the same CoT prompt over all accumulated paragraphs, answer = the text after "answer is".
Differences (disclosed in the paper): BM25 from bm25s over the same corpora instead of Elasticsearch; the backbone is a chat
model used through the plain completion endpoint; prompt length is estimated as characters / 4 instead of GPT-2 tokens;
sentences are split with a regex instead of spaCy. Paragraphs are truncated to 350 words as in the official code.
"""
import json
import os
import re
from pathlib import Path

from ..utils.env import ROOT

# unpacked cache/ircot/downloads/ircot_repo.tar.gz; IRCOT_REPO overrides it for local tests
IRCOT_REPO = Path(os.environ.get("IRCOT_REPO", ROOT / "cache" / "ircot" / "ircot_repo"))
MAX_PARAS, MAX_SENTS, MAX_PARA_WORDS = 15, 10, 350
LENGTH_LIMIT, GEN_RESERVE = 8000, 300
ANSWER_RE = re.compile(r".* answer is:? (.*)", flags=re.S)

# valid_qids of base_configs/ircot_qa_codex_<ds>.jsonnet (the demonstrations and their order)
VALID_QIDS = {
    "hotpotqa": ["5ab92dba554299131ca422a2", "5a7bbc50554299042af8f7d0", "5add363c5542990dbb2f7dc8",
                 "5a835abe5542996488c2e426", "5ae0185b55429942ec259c1b", "5a790e7855429970f5fffe3d",
                 "5a754ab35542993748c89819", "5a89c14f5542993b751ca98a", "5abb14bd5542992ccd8e7f07",
                 "5a89d58755429946c8d6e9d9", "5a88f9d55542995153361218", "5a90620755429933b8a20508",
                 "5a77acab5542992a6e59df76", "5abfb3435542990832d3a1c1", "5a8f44ab5542992414482a25",
                 "5adfad0c554299603e41835a", "5a7fc53555429969796c1b55", "5a8ed9f355429917b4a5bddd",
                 "5ac2ada5554299657fa2900d", "5a758ea55542992db9473680"],
    "2wikimultihopqa": ["5811079c0bdc11eba7f7acde48001122", "97954d9408b011ebbd84ac1f6bf848b6",
                        "35bf3490096d11ebbdafac1f6bf848b6", "c6805b2908a911ebbd80ac1f6bf848b6",
                        "5897ec7a086c11ebbd61ac1f6bf848b6", "e5150a5a0bda11eba7f7acde48001122",
                        "a5995da508ab11ebbd82ac1f6bf848b6", "cdbb82ec0baf11ebab90acde48001122",
                        "f44939100bda11eba7f7acde48001122", "4724c54e08e011ebbda1ac1f6bf848b6",
                        "f86b4a28091711ebbdaeac1f6bf848b6", "13cda43c09b311ebbdb0ac1f6bf848b6",
                        "228546780bdd11eba7f7acde48001122", "c6f63bfb089e11ebbd78ac1f6bf848b6",
                        "1ceeab380baf11ebab90acde48001122", "8727d1280bdc11eba7f7acde48001122",
                        "f1ccdfee094011ebbdaeac1f6bf848b6", "79a863dc0bdc11eba7f7acde48001122",
                        "028eaef60bdb11eba7f7acde48001122", "af8c6722088b11ebbd6fac1f6bf848b6"],
    "musique": ["2hop__323282_79175", "2hop__292995_8796", "2hop__439265_539716", "4hop3__703974_789671_24078_24137",
                "2hop__154225_727337", "2hop__861128_15822", "3hop1__858730_386977_851569", "2hop__642271_608104",
                "2hop__387702_20661", "2hop__131516_53573", "2hop__496817_701819", "2hop__804754_52230",
                "3hop1__61746_67065_43617", "3hop1__753524_742157_573834", "2hop__427213_79175",
                "3hop1__443556_763924_573834", "2hop__782642_52667", "2hop__102217_58400", "2hop__195347_20661",
                "4hop3__463724_100414_35260_54090"],
}
_WH = {"who", "what", "when", "where", "why", "which", "how", "does", "is"}
_REASONING_STARTS = ("thus ", "thus,", "so ", "so,", "that is,", "therefore", "hence")
_ARITH = re.compile(r"(.*)(\d[\d,]*\.?\d+|\d+) ([+-]) (\d[\d,]*\.?\d+|\d+) = (\d[\d,]*\.?\d+|\d+)(.*)")
_DEMOS = {}


def demonstrations(ds, n_demos=None):
    """The CoT demonstrations of the official prompt file, filtered, ordered and length-limited like read_prompt();
    n_demos keeps only the first n (budget-reduced variants for the quality-cost curve, not the official setting)."""
    if (ds, n_demos) in _DEMOS:
        return _DEMOS[(ds, n_demos)]
    path = IRCOT_REPO / "prompts" / ds / "gold_with_2_distractors_context_cot_qa_codex.txt"
    blocks, cur, qid = {}, [], None
    for line in path.read_text().strip().split("\n"):
        if line.strip().startswith("# METADATA: "):
            if qid is not None:
                blocks[qid] = "\n".join(cur).strip()
            qid, cur = json.loads(line.strip()[len("# METADATA: "):])["qid"], []
        else:
            cur.append(line)
    if qid is not None:
        blocks[qid] = "\n".join(cur).strip()
    demos = [blocks[q] for q in VALID_QIDS[ds] if q in blocks]
    lens = [len(d) // 4 for d in demos]
    while demos and sum(lens) + max(lens) + GEN_RESERVE > LENGTH_LIMIT:
        demos.pop(); lens.pop()
    if n_demos is not None:
        demos = demos[:n_demos]
    _DEMOS[(ds, n_demos)] = "\n\n\n".join(demos)
    return _DEMOS[(ds, n_demos)]


def remove_wh_words(text):
    return " ".join(w for w in text.split(" ") if w.strip().lower() not in _WH)


def is_reasoning_sentence(s):
    return s.lower().startswith(_REASONING_STARTS) or bool(_ARITH.match(s))


def first_sentence(text):
    text = text.strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text, maxsplit=1)
    return parts[0].strip() if parts else ""


def para_text(title, text):
    return f"Wikipedia Title: {title}\n" + " ".join(text.split(" ")[:MAX_PARA_WORDS]).strip()


def extract_answer(text):
    m = ANSWER_RE.match(text.strip())
    ans = m.group(1) if m else text.strip()
    ans = ans.strip().split("\n")[0].strip()
    return ans[:-1] if ans.endswith(".") else ans


class IRCoT:
    def __init__(self, llm, index, ds, k=6, n_demos=None, max_paras=MAX_PARAS):
        """n_demos / max_paras below the official values give the budget-reduced points of IRCoT's quality-cost curve."""
        self.llm, self.index, self.ds, self.k, self.max_paras = llm, index, ds, k, max_paras
        self.demos = demonstrations(ds, n_demos)

    def _add(self, paras, query):
        q = remove_wh_words(query)
        if not q.strip():
            return
        for h in self.index.search(q, k=self.k):
            if len(paras) >= self.max_paras:
                break
            if any(h["title"] == t and h["text"] == x for t, x in paras):
                continue
            if len(h["text"].split(" ")) > 600:
                continue
            paras.append((h["title"], h["text"]))

    def _prompt(self, paras, question, cot):
        ctx = "\n\n".join(para_text(t, x) for t, x in paras)
        return "\n\n\n".join([self.demos, f"{ctx}\n\nQ: {question}\nA: {cot}".rstrip()]).strip()

    def run(self, q):
        question, paras, sents = q["question"], [], []
        tok = calls = 0
        self._add(paras, question)
        for _ in range(MAX_SENTS):
            out = self.llm.complete(self._prompt(paras, question, " ".join(sents)), temperature=0.0, max_tokens=200,
                                    stop=["\n"])
            tok += out["usage"]["prompt"] + out["usage"]["completion"]; calls += 1
            sent = first_sentence(out["texts"][0])
            if not sent:
                break
            sents.append(sent)
            if ANSWER_RE.match(sent):
                break
            facts = [s for s in sents if not is_reasoning_sentence(s)]
            self._add(paras, facts[-1] if facts else question)
        out = self.llm.complete(self._prompt(paras, question, ""), temperature=0.0, max_tokens=300, stop=["\n\n\n", "\nQ:"])
        tok += out["usage"]["prompt"] + out["usage"]["completion"]; calls += 1
        return {"pred": extract_answer(out["texts"][0]), "tok": tok, "calls": calls, "n_paras": len(paras),
                "cot": " ".join(sents), "reader": out["texts"][0][:500]}
