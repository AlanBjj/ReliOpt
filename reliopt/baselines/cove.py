"""Chain-of-Verification, factored variant (Dhuliawala et al., Findings of ACL 2024), after the prompt templates of the
paper's appendix B (arXiv 2309.11495v2, appendix.tex). No code was released; the templates there are schematic and each task
uses 3 examples of its own, so the 3 examples below are written in the same fictional style as this project's planner.

Steps: (1) baseline response = the Standard RAG draft (same prompt and greedy call as reliopt.baselines.simple, so the
call is shared with that baseline); (2) plan verifications: "Context: Q: <question> A: <draft> Response:" -> lines
"<fact in the draft>, <verification question>"; (3) execute each verification question independently (factored): the
paper answers them closed-book, here they are answered by the BM25 reader shared with Self-Ask (top-5 passages, the default
step reader of reliopt.exec.steps); (4) final verified response: "Context: ... From another source, <Q + A> Response:".
All template calls are greedy, through the plain completion endpoint (as the few-shot templates are written).
Differences (disclosed): retrieval-backed verification answers; at most MAX_VQ verification questions; the final response
ends with "Answer: <short answer>" so it is scored like every other method (the draft's answer if it does not).
"""
import re

from ..exec.steps import StepExecutor, parse_answer
from .simple import MAX_TOKENS, rag_messages

MAX_VQ = 5

PLAN_EXAMPLES = """Context: Q: In which city was the director of the film Northern Shore born?
A: Northern Shore is a 1987 drama film directed by Lena Varga. Lena Varga was born in Szeged. Answer: Szeged
Response:
Northern Shore was directed by Lena Varga, Who directed the film Northern Shore?
Lena Varga was born in Szeged, In which city was Lena Varga born?

Context: Q: Which company was founded earlier, Arlen Motors or Bexley Foods?
A: Arlen Motors was founded in 1921 and Bexley Foods in 1930, so Arlen Motors was founded earlier. Answer: Arlen Motors
Response:
Arlen Motors was founded in 1921, When was Arlen Motors founded?
Bexley Foods was founded in 1930, When was Bexley Foods founded?

Context: Q: Are the authors of the novels Grey Harbor and The Tin Orchard of the same nationality?
A: Grey Harbor was written by Tomas Reyes, who is Chilean. The Tin Orchard was written by Ada Brandt, who is German. They are not of the same nationality. Answer: no
Response:
Grey Harbor was written by Tomas Reyes, Who wrote the novel Grey Harbor?
Tomas Reyes is Chilean, What is the nationality of Tomas Reyes?
The Tin Orchard was written by Ada Brandt, Who wrote the novel The Tin Orchard?
Ada Brandt is German, What is the nationality of Ada Brandt?

"""

FINAL_EXAMPLES = """Context: Q: In which city was the director of the film Northern Shore born?
A: Northern Shore is a 1987 drama film directed by Lena Varga. Lena Varga was born in Budapest. Answer: Budapest
From another source,
Q: Who directed the film Northern Shore? A: Lena Varga
Q: In which city was Lena Varga born? A: Szeged
Response: Northern Shore was directed by Lena Varga, who was born in Szeged. Answer: Szeged

Context: Q: Which company was founded earlier, Arlen Motors or Bexley Foods?
A: Arlen Motors was founded in 1921 and Bexley Foods in 1930, so Arlen Motors was founded earlier. Answer: Arlen Motors
From another source,
Q: When was Arlen Motors founded? A: 1921
Q: When was Bexley Foods founded? A: 1930
Response: Arlen Motors was founded in 1921, before Bexley Foods in 1930. Answer: Arlen Motors

Context: Q: Are the authors of the novels Grey Harbor and The Tin Orchard of the same nationality?
A: Grey Harbor was written by Tomas Reyes, who is Chilean. The Tin Orchard was written by Ada Brandt, who is Chilean. They are of the same nationality. Answer: yes
From another source,
Q: Who wrote the novel Grey Harbor? A: Tomas Reyes
Q: What is the nationality of Tomas Reyes? A: Chilean
Q: Who wrote the novel The Tin Orchard? A: Ada Brandt
Q: What is the nationality of Ada Brandt? A: German
Response: Tomas Reyes is Chilean and Ada Brandt is German, so they are not of the same nationality. Answer: no

"""

VQ_LINE = re.compile(r"^(.*),\s*([^,]+\?)\s*$")


def one_line(text):
    return " ".join(text.split())


class CoVe:
    def __init__(self, llm, index, k=5):
        self.llm, self.index, self.k = llm, index, k
        self.reader = StepExecutor(llm, index)
        self.tok = self.calls = 0

    def _count(self, out):
        self.tok += out["usage"]["prompt"] + out["usage"]["completion"]
        self.calls += 1
        return out["texts"][0]

    def run(self, q):
        question = q["question"]
        draft = self._count(self.llm.chat(rag_messages(question, self.index.search(question, k=self.k)),
                                          temperature=0.0, max_tokens=MAX_TOKENS))
        draft_answer = parse_answer(draft)
        context = f"Context: Q: {question}\nA: {one_line(draft)}\n"
        plan = self._count(self.llm.complete(PLAN_EXAMPLES + context + "Response: \n", temperature=0.0, max_tokens=256,
                                             stop=["\n\n", "\nContext:"]))
        vqs = []
        for line in plan.strip().splitlines():
            m = VQ_LINE.match(line.strip())
            if m and m.group(2).strip() not in vqs:
                vqs.append(m.group(2).strip())
        vqs = vqs[:MAX_VQ]
        qa = []
        for vq in vqs:   # factored: each verification question is answered on its own
            hits = self.index.search(vq, k=self.k)
            ans = parse_answer(self._count(self.llm.chat(self.reader.txt_messages(vq, hits), temperature=0.0,
                                                         max_tokens=64)))
            qa.append(f"Q: {vq} A: {ans}")
        if not qa:
            return {"pred": draft_answer, "tok": self.tok, "calls": self.calls, "n_vq": 0, "revised": False}
        final = self._count(self.llm.complete(FINAL_EXAMPLES + context + "From another source, \n" + "\n".join(qa)
                                              + "\nResponse:", temperature=0.0, max_tokens=256, stop=["\n"]))
        m = re.search(r"Answer\s*:\s*(.+)$", final.strip())
        pred = m.group(1).strip().rstrip(".") if m else draft_answer
        return {"pred": pred, "tok": self.tok, "calls": self.calls, "n_vq": len(vqs), "revised": pred != draft_answer,
                "verifications": qa, "final": final.strip()}
