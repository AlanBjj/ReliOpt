"""Self-Ask with search (Press et al., Findings of EMNLP 2023), after the official demo (ofirpress/self-ask, notebook
self-ask_plus_search-engine_demo.ipynb): the 4-shot prompt and the control loop of promptf() are copied.

The search engine's answer box is replaced by BM25 (top-5 over the dataset corpus) read by the same backbone with the step
reader used everywhere in this project (reliopt.exec.steps TXT prompt); when the reader finds nothing, the model writes
the intermediate answer itself, as the official code does when Google returns no answer. Generation is greedy through the
plain completion endpoint with the official stop strings; max_tokens 256 as in call_gpt(). Follow-ups are capped at 8.
Two fixes to the demo code: after a self-written intermediate answer the model is asked to continue (the demo re-enters the
loop with the old text), and an answer containing a colon is kept whole (see extract_answer).
"""
from ..exec.steps import StepExecutor, is_abstain

PROMPT = ['''Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali

Question: When was the founder of craigslist born?
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952

Question: Who was the maternal grandfather of George Washington?
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball

Question: Are both the directors of Jaws and Casino Royale from the same country?
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate Answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate Answer: The United States.
Follow up: Who is the director of Casino Royale?
Intermediate Answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate Answer: New Zealand.
So the final answer is: No

Question: ''', '''
Are follow up questions needed here:''']
INTERMEDIATE, FOLLOWUP, FINALANS = "\nIntermediate answer:", "Follow up:", "\nSo the final answer is:"
# The official stop is "\nIntermediate answer:", but the prompt's fourth demonstration writes "Intermediate Answer:", which
# Llama copies; without the capitalised stop the model writes its own answers and runs on into invented new questions.
STOP_INTERMEDIATE = [INTERMEDIATE, "\nIntermediate Answer:", "\nQuestion:"]
MAX_TOKENS, MAX_FOLLOWUPS = 256, 8


def _last_line(text):
    return text.split("\n")[-1] if "\n" in text else text


def extract_answer(generated):
    """Official extract_answer() (text after the last colon of the last line, no trailing full stop), except that the text
    after "the final answer is:" is kept whole, so answers that contain a colon ("Danger: Diabolik") are not cut."""
    marker = "the final answer is:"
    i = generated.lower().rfind(marker)
    if i >= 0:
        after = generated[i + len(marker):].strip().split("\n")[0]
    else:
        after = generated.split(":")[-1] if ":" in _last_line(generated) else _last_line(generated)
    after = after.strip()
    return after[:-1] if after.endswith(".") else after


def extract_question(generated):
    last = _last_line(generated)
    after = last.split(":")[-1] if ":" in last else last
    return after.strip()


class SelfAsk:
    def __init__(self, llm, index):
        self.llm = llm
        self.reader = StepExecutor(llm, index)

    def _gen(self, prompt, stop):
        out = self.llm.complete(prompt, temperature=0.0, max_tokens=MAX_TOKENS, stop=stop)
        self.tok += out["usage"]["prompt"] + out["usage"]["completion"]; self.calls += 1
        return out["texts"][0]

    def _search(self, question):
        hits = self.reader.retrieve(question)
        out = self.llm.chat(self.reader.txt_messages(question, hits), temperature=0.0, max_tokens=64)
        self.tok += out["usage"]["prompt"] + out["usage"]["completion"]; self.calls += 1
        ans = self.reader._txt_cand(out["texts"][0], None, None, hits)["answer"]
        return None if is_abstain(ans) else ans

    def run(self, q):
        self.tok = self.calls = 0
        cur = PROMPT[0] + q["question"] + PROMPT[1]
        ret = self._gen(cur, STOP_INTERMEDIATE)
        n = 0
        while FOLLOWUP in _last_line(ret) and n < MAX_FOLLOWUPS:
            n += 1
            cur += ret
            ext = self._search(extract_question(ret))
            if ext is not None:
                cur += INTERMEDIATE + " " + ext + "."
                ret = self._gen(cur, STOP_INTERMEDIATE)
            else:
                cur += INTERMEDIATE
                cur += self._gen(cur, ["\n" + FOLLOWUP, FINALANS, "\nQuestion:"])
                ret = self._gen(cur, STOP_INTERMEDIATE)
        if FINALANS not in ret:
            cur += ret + FINALANS
            ret = self._gen(cur, ["\n"])
        return {"pred": extract_answer(ret), "tok": self.tok, "calls": self.calls, "followups": n,
                "trace": (cur + ret)[len(PROMPT[0]):][-1500:]}
