"""Plan-and-Budget (Lin et al., ICLR 2026), ported from the official code (junhongmit/P-and-B, commit 2ca2d90):
dataset/break_down_question.py (decomposition and credit assessment), inference/_5_planned_local_weighted_model.py (budgeted
solving), utils/utils.py (level_map, sampling defaults), dataset/instruction_dataset.py (instruction and output format of the
non-math task) and utils/qwen_math_parser.py (answer extraction). Prompts are copied verbatim, including the literal
"\\boxed{{answer}}" that the official .format() call leaves in the output format.

Pipeline (three calls): break the question down into 2-5 numbered sub-questions with hints -> assess the problem's level
(1-5) and each sub-question's credit (JSON mode) -> one solving call in which sub-question i is told "Please only think a
little, and directly solve it using up to b_i words.", with b = allocate_tokens(credits, level_map[level], schedule) and the
polynomial decay schedule (N - i)^2 (the local weighted variant the paper recommends for complex reasoning); the answer is the
last \\boxed{...}. Official sampling: temperature 0.1, top-p 0.9, max_tokens 8192 for every call; failed JSON is retried.
Differences (disclosed):
  - Multi-hop QA needs evidence, which the method itself never retrieves. BM25 paragraphs are put into the prompt's
    Reference field (the slot the official prompt fills with a reference text, "None." for MATH). retrieval="question":
    top-5 for the question (the Standard RAG context); retrieval="subq": also top-5 for each sub-question, deduplicated, at
    most 15 paragraphs. The setting is chosen per dataset on the tuning split.
  - No difficulty labels: the level is the planner's own "problem" level from the assessment call (the official runs use
    the MATH label or levels shipped with the data); the decomposition prompt therefore has no "Level:" line.
  - The domain string is "multi-hop question answering" (the official code passes "math", "task completions" or
    "travel planning" into the same math-worded prompt); the benchmark examples are "None", as for NaturalInstructions.
  - The answer is cleaned with the generic parts of strip_string only (line breaks, trailing ".", $ signs, \\text{});
    its unit-word and number-word rewriting is math-specific and would delete words such as "years" from answers.
  - Llama-3.1-8B-Instruct is a chat model, so the budget is only the words requested in the prompt plus the max_tokens
    cap; with Qwen3-8B the solving call runs in thinking mode (pass the qwen3-8b-think client as `solver`).
"""
import json
import re
import textwrap

TEMPERATURE, TOP_P, MAX_TOKENS = 0.1, 0.9, 8192
TOPK, MAX_PARAS, MAX_RETRIES = 5, 15, 3
DOMAIN = "multi-hop question answering"
LEVEL_MAP = {1: ("simple", 200), 2: ("simple", 250), 3: ("medium", 350), 4: ("medium", 450), 5: ("hard", 600)}

BREAK_DOWN_SYSTEM = textwrap.dedent("""\
    -Goal-
    You are an experienced expert in {domain} and exam question designer. Your role is to help students break down challenging math problems into a series of simpler, high-level sub-questions.
    We don't want too many detailed sub-questions, which are not beneficial for testing students' ability in an exam. Each sub-question should build on the previous one so that, once all have been answered, the complete solution is clear.
    Your output should be a list of sub-questions with brief hints explaining the purpose of each step, but you should not reveal your internal chain-of-thought either the final solution.

    Instructions for Decomposition:
    First, analyze the problem and identify the key ideas needed to solve it. Then, generate a series of 2 to 5 sub-questions that lead the student step by step to the complete solution.
    The difficulty level of the problem is presented out of 5, wher 1 is easy, and 5 is hard. Please adjust the number of sub-questions based on the level. Ideally, we want fewer sub-questions for
    easy problems and more sub-questions for challenging problems.
    DO NOT perform reasoning, directly output those sub-questions based on your gut feelings; only output the list of sub-questions with brief hints for each.
    Your answer should be a list of numbered sub-questions. Each sub-question should have a brief accompanying hint that explains what the student will achieve by answering that part.\x20

    Example Decomposition:
    **Problem:** Find the remainder when \\(9 \\times 99 \\times 999 \\times \\cdots \\times \\underbrace{{99\\cdots9}}_{{\\text{{999 9's}}}}\\) is divided by 1000.
    **Level:** 3 out of 5

    **Decomposed Sub-questions:**

    1. Compute the product modulo 8.
    Hint: Simplify each term using \\(10 \\equiv 2 \\mod 8\\), noting that \\(10^k \\equiv 0 \\mod 8\\) for \\(k \\geq 3\\), leading to terms of \\(-1 \\mod 8\\).

    2. Compute the product modulo 125.
    Hint: Recognize \\(10^3 \\equiv 0 \\mod 125\\), so terms for \\(k \\geq 3\\) become \\(-1 \\mod 125\\). Calculate the product of the first two terms and combine with the remaining terms.

    3. Solve the system of congruences using the Chinese Remainder Theorem.
    Hint: Combine the results from modulo 8 and modulo 125 to find a common solution modulo 1000.
""")
# official user message minus its "Level: {level} out of 5" line (no difficulty label here, see the module docstring)
BREAK_DOWN_USER = textwrap.dedent("""\
    A student has presented you with the following math problem:
    Problem: {problem}
    **REMEMBER**, you are not allowed to think about it, please directly generate the answer in the following:
    Decomposed Sub-questions:
""")
ASSESS_SYSTEM = textwrap.dedent("""\
    You are an experienced expert in {domain} and exam question designer. Your task is to evaluate the difficulty level of a given exam problem and its sub-questions by comparing it against a set of benchmark questions of known levels.
    Based on their levels, you will need to assign each subquestion a portion of the credits (assuming the total credit points is 100 for the whole problem).

    Each level reflects increasing complexity from 1 (easiest) to 5 (most challenging). Evaluate based on the conceptual depth, steps involved in solving, required knowledge, and potential for misdirection.

    Use the following benchmark examples as references:

    {benchmarks}

    1. You will be provided a question and its subquestions. You will evaluate the difficulty level of the problem and its sub-questions.
    Assuming the whole problem is worth 100 points, you assign each sub-question a portion of the score points.
    - Adhere to the given subquestions, and DO NOT make new subquestions.
    - Sum of each subquestion's credits MUST EQUAL to 100.

    2. You must return the result in a structured JSON format:
    {{
    "problem": {{"reason": "...", "evaluated_level": level_q}}
    "1": {{"reason": "...", "evaluated_level": level_1, "credit": credit_1}},
    "2": {{"reason": "...", "evaluated_level": level_2, "credit": credit_2}},
    ...}}
    where
    - "reason": a short explanation (up to 50 words) of your level assessment.
    - "evaluated_level": an integer from 1 to 5 indicating your judgment.
    - "credit": an integer between 1 to 100 indicating when the question is solved correctly, how many credit can be given.
""")
ASSESS_USER = textwrap.dedent("""\
    Evaluate the level of the following question:
    Problem: {problem}
    Sub-questions: {steps}
    Output:""")
SOLVE_SYSTEM = textwrap.dedent("""\
    {instruction}

    The problem is given by an overall description, difficulty level out of 5, followed by a series of sub-questions as a hint.
    All the credit is given when you provide a correct final answer for the overall problem.
    Please solve the question efficiently and clearly to achieve as much credit as possible.""")
SOLVE_USER = textwrap.dedent("""\
    Let's start the exam. You are being given this math problem:
    **Problem (100pt):** {question}
    **Reference:** {reference}
    **Level:** {level} out of 5

    You may think following these sub-questions or feel free to use other methods that works the best towards getting the final answer:
    {decomposed}

    Please provide your final answer strictly following the format:
    {output_format}

    Output: <think>\n""")
INSTRUCTION = "You are a student being tested on your problem-solving skills in an exam."
OUTPUT_FORMAT = textwrap.dedent("""\
    **Final Answer:** Therefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.

    Where [answer] is just the final number or expression in latex format that solves the problem.
    If you have a list of answers, simply concat them into a comma-sparated list, for example: $\\boxed{{1, \\sqrt{{2}}, 3}}$.
    """)
STEP_RE = re.compile(r"(\d+\..*?)(?=\nHint:)(?:\nHint:\s*(.*?))(?:\n|$)", flags=re.S)


def polynomial_decay(i, n, power=2):
    return (n - i) ** power


def allocate_tokens(credits, total, schedule=polynomial_decay):
    """Official allocate_tokens: credit x schedule weights, normalised, floored, remainder to the largest shares."""
    n = len(credits)
    combined = [c * schedule(i, n) for i, c in enumerate(credits)]
    s = sum(combined)
    combined = [x / s for x in combined] if s > 0 else [1 / n] * n
    tokens = [int(x * total) for x in combined]
    for i in sorted(range(n), key=lambda j: -combined[j])[:total - sum(tokens)]:
        tokens[i] += 1
    return tokens


def parse_steps(text):
    """Official parsing of the decomposition: '<n>. sub-question' followed by a 'Hint:' line."""
    steps = [f"{q} Hint: {h.strip()}" for q, h in STEP_RE.findall(text)]
    return steps or ["1. Directly solve the problem. Hint: None."]


def step_query(step):
    """Retrieval query of a sub-question: its text without the number and the hint."""
    return re.sub(r"^\s*\d+\.\s*", "", step.split(" Hint:")[0]).strip()


def load_json(text):
    """maybe_load_json: the whole text, else the first {...} object in it."""
    try:
        return json.loads(text)
    except ValueError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for j in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                try:
                    return json.loads(text[start:j + 1])
                except ValueError:
                    break
        start = text.find("{", start + 1)
    return None


def extract_answer(pred):
    """Official extract_answer(pred, 'instruction') followed by the generic part of strip_string."""
    pred = pred.replace("\u043a\u0438", "")
    if "final answer is $" in pred and "$. I hope" in pred:
        a = pred.split("final answer is $", 1)[1].split("$. I hope", 1)[0].strip()
    elif "boxed" in pred:
        ans = pred.split("boxed")[-1]
        if not ans:
            a = ""
        elif ans[0] == "{":
            depth, a = 1, ""
            for c in ans[1:]:
                depth += {"{": 1, "}": -1}.get(c, 0)
                if depth == 0:
                    break
                a += c
        else:
            a = ans.split("$")[0].strip()
    elif "he answer is" in pred:
        a = pred.split("he answer is")[-1].strip()
    elif "final answer is" in pred:
        a = pred.split("final answer is")[-1].strip()
    else:
        nums = re.findall(r"-?\d*\.?\d+", pred.replace(",", ""))
        a = nums[-1] if nums else ""
    a = a.strip().replace("\n", "").rstrip(".")
    m = re.fullmatch(r"\\text\{(.*)\}", a)
    if m:
        a = m.group(1)
    return a.replace("\\$", "").replace("$", "").replace("\\(", "").replace("\\)", "").strip()


class PlanAndBudget:
    def __init__(self, llm, index, seed=17, retrieval="question", solver=None):
        assert retrieval in ("question", "subq"), retrieval
        self.llm, self.index, self.seed, self.retrieval = llm, index, seed, retrieval
        self.solver = solver or llm
        self.tok = self.calls = 0

    def _chat(self, llm, msgs, seed, **kw):
        out = llm.chat(msgs, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS, seed=seed, **kw)
        self.tok += out["usage"]["prompt"] + out["usage"]["completion"]
        self.calls += 1
        return out

    def plan(self, question):
        msgs = [{"role": "system", "content": BREAK_DOWN_SYSTEM.format(domain=DOMAIN)},
                {"role": "user", "content": BREAK_DOWN_USER.format(problem=question)}]
        steps = parse_steps(self._chat(self.llm, msgs, self.seed)["texts"][0])
        msgs = [{"role": "system", "content": ASSESS_SYSTEM.format(domain=DOMAIN, benchmarks="None")},
                {"role": "user", "content": ASSESS_USER.format(problem=question, steps=steps)}]
        for attempt in range(MAX_RETRIES):
            res = load_json(self._chat(self.llm, msgs, self.seed + attempt,
                                       extra={"response_format": {"type": "json_object"}})["texts"][0])
            try:
                level = min(5, max(1, int(res["problem"]["evaluated_level"])))
                credits = [float(res[str(i + 1)]["credit"]) for i in range(len(steps))]
                return steps, level, credits, False
            except (TypeError, KeyError, ValueError):
                continue
        return steps, 3, [1.0] * len(steps), True   # assessment never parsed: middle level, equal credits

    def references(self, question, steps):
        paras, seen = [], set()
        queries = [question] + ([step_query(s) for s in steps] if self.retrieval == "subq" else [])
        for q in queries:
            for h in self.index.search(q, k=TOPK):
                if h["id"] not in seen and len(paras) < MAX_PARAS:
                    seen.add(h["id"])
                    paras.append(h)
        return "\n".join(f"[{i}] {h['title']}: {h['text']}" for i, h in enumerate(paras, 1)) or "None.", len(paras)

    def run(self, q):
        steps, level, credits, fallback = self.plan(q["question"])
        budgets = allocate_tokens(credits, LEVEL_MAP[level][1])
        decomposed = "\n\n".join(f"{s} Please only think a little, and directly solve it using up to {b} words."
                                 for s, b in zip(steps, budgets))
        reference, n_paras = self.references(q["question"], steps)
        msgs = [{"role": "system", "content": SOLVE_SYSTEM.format(instruction=INSTRUCTION)},
                {"role": "user", "content": SOLVE_USER.format(question=q["question"], reference=reference, level=level,
                                                              decomposed=decomposed, output_format=OUTPUT_FORMAT)}]
        out = self._chat(self.solver, msgs, self.seed)
        return {"pred": extract_answer(out["texts"][0]), "tok": self.tok, "calls": self.calls, "level": level,
                "steps": steps, "budgets": budgets, "paragraphs": n_paras, "assess_fallback": fallback,
                "output": out["texts"][0][-2000:]}
