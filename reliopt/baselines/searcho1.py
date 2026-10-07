"""Search-o1 (Li et al., EMNLP 2025), ported from the official code (RUC-NLPIR/Search-o1, commit c76a700, MIT licence):
scripts/run_search_o1.py and scripts/prompts.py; the instructions are copied verbatim.

The model reasons and writes <|begin_search_query|> q <|end_search_query|> when it needs knowledge; generation stops there,
the query is searched (top-10), and the Reason-in-Documents call reads the documents against the previous reasoning and
returns "**Final Information**", which is appended as <|begin_search_result|> ... <|end_search_result|>; reasoning then
continues. Multi-hop QA settings of the official script: at most 10 searches and 15 turns, top-10 documents, sampling
temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.0 (1.05 in Reason-in-Documents); the answer is the last
\\boxed{...}. As in the original, the user turn is rendered with the model's chat template (apply_chat_template, generation
prompt opened) and the growing assistant text is appended and continued with plain completion (prompt += text).
Differences (disclosed): BM25 paragraphs of the IRCoT corpora replace Bing results and fetched web pages (each document is
shown as {"title", "context"}); max_tokens per call is 4096 instead of 8192 to stay within the 16k context.
(A first version continued with vLLM's continue_final_message; Llama's template trims the trailing newlines, the model then
ended its turn right after a search result, and 145 of 200 2Wiki pilot answers were empty.)
"""
import json
import re

from ..llm.template import render

BEGIN_Q, END_Q = "<|begin_search_query|>", "<|end_search_query|>"
BEGIN_R, END_R = "<|begin_search_result|>", "<|end_search_result|>"
MAX_SEARCH, MAX_TURN, TOPK = 10, 15, 10
MAX_TOKENS, MAX_OUTPUT_CHARS = 4096, 40000
SAMPLING = {"temperature": 0.7, "top_p": 0.8}


def multiqa_instruction(max_search=MAX_SEARCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {max_search}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def task_instruction_openqa(question):
    return ('Please answer the following question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n')


def webpage_to_reasonchain_instruction(prev_reasoning, search_query, document):
    return f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**
{prev_reasoning}

- **Current Search Query:**
{search_query}

- **Searched Web Pages:**
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""


def extract_between(text, start, end):
    m = re.findall(re.escape(start) + r"(.*?)" + re.escape(end), text, flags=re.S)
    return m[-1].strip() if m else None


def extract_info(output):
    """Official extract_answer(mode='infogen')."""
    if "**Final Information**" in output:
        return output.split("**Final Information**")[-1].replace("\n", "").strip("```").strip()
    if "**Modified Reasoning Steps**" in output:
        return output.split("**Modified Reasoning Steps**")[-1].strip("```").strip()
    return "No helpful information found."


def extract_boxed(output):
    """Official extract_answer(mode='qa'): last \\boxed{...}, inner \\text{...} unwrapped, parentheses stripped."""
    m = re.findall(r"\\boxed\{(.*)\}", output)
    if not m:
        return ""
    ans = m[-1]
    inner = re.findall(r"\\text\{(.*)\}", ans)
    if inner:
        ans = inner[-1]
    return ans.strip("()")


def truncated_reasoning(output):
    """Official truncation of the previous reasoning passed to Reason-in-Documents."""
    steps = output.replace("\n\n", "\n").split("\n")
    text = "".join(f"Step {i + 1}: {s}\n\n" for i, s in enumerate(steps))
    prev = text.split("\n\n")
    if len(prev) <= 5:
        return "\n\n".join(prev).strip("\n")
    out = ""
    for i, s in enumerate(prev):
        if i == 0 or i >= len(prev) - 4 or BEGIN_Q in s or BEGIN_R in s:
            out += s + "\n\n"
        elif out[-len("\n\n...\n\n"):] != "\n\n...\n\n":
            out += "...\n\n"
    return out.strip("\n")


class SearchO1:
    def __init__(self, llm, index, seed=17):
        self.llm, self.index, self.seed = llm, index, seed

    def _chat(self, msgs, extra, stop=None):
        out = self.llm.chat(msgs, max_tokens=MAX_TOKENS, seed=self.seed, stop=stop, extra=extra, **SAMPLING)
        self.tok += out["usage"]["prompt"] + out["usage"]["completion"]; self.calls += 1
        return out["texts"][0]

    def _continue(self, prompt):
        out = self.llm.complete(prompt, max_tokens=MAX_TOKENS, seed=self.seed, stop=[END_Q], **SAMPLING,
                                extra={"top_k": 20, "repetition_penalty": 1.0, "include_stop_str_in_output": True,
                                       "add_special_tokens": False})
        self.tok += out["usage"]["prompt"] + out["usage"]["completion"]; self.calls += 1
        return out["texts"][0]

    def run(self, q):
        self.tok = self.calls = 0
        user = multiqa_instruction() + task_instruction_openqa(q["question"])
        prompt = render(self.llm.tag, [{"role": "user", "content": user}])
        output, n_search, done = "", 0, set()
        for _ in range(MAX_TURN):
            text = self._continue(prompt + output)
            output += text
            query = extract_between(text, BEGIN_Q, END_Q)
            if not (query and output.rstrip().endswith(END_Q)) or len(output) > MAX_OUTPUT_CHARS:
                break
            if n_search < MAX_SEARCH and query not in done:
                hits = self.index.search(query, k=TOPK)
                docs = "".join(f"**Web Page {i + 1}:**\n" + json.dumps({"title": h["title"], "context": h["text"]},
                                                                       ensure_ascii=False, indent=2) + "\n"
                               for i, h in enumerate(hits))
                raw = self._chat([{"role": "user", "content":
                                   webpage_to_reasonchain_instruction(truncated_reasoning(output), query, docs)}],
                                 {"top_k": 20, "repetition_penalty": 1.05})
                output += f"\n\n{BEGIN_R}{extract_info(raw)}{END_R}\n\n"
                n_search += 1
                done.add(query)
            elif n_search >= MAX_SEARCH:
                output += f"\n{BEGIN_R}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_R}\n"
            else:
                output += f"\n{BEGIN_R}\nYou have searched this query. Please refer to previous results.\n{END_R}\n"
        return {"pred": extract_boxed(output), "tok": self.tok, "calls": self.calls, "searches": n_search,
                "output": output[-1500:]}
