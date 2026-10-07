"""Fake OpenAI-compatible chat server for local tests. Behaviour is chosen by a responder function on the request.

Usage: python tests/fake_vllm.py PORT [mode]   (mode "echo" or "e1"; tests/toy_e1.sh runs it on port 8199)
Counts requests in /stats.
"""
import json
import random
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATS = {"requests": 0}
LOCK = threading.Lock()


def tokens_of(text):
    return re.findall(r"\s*\S+|\s+", text) or [""]


def echo_responder(req, i, rng):
    time.sleep(0.2)
    return "Answer: Jane Austen\nPassages: 1"


def completion_responder(req, i, rng):
    prompt = req["prompt"]
    if prompt.rstrip().endswith("Are follow up questions needed here:"):
        return " Yes.\nFollow up: Who directed Gamma Film?"
    if "Intermediate answer:" in prompt[-200:] and "So the final answer is:" not in prompt[-60:]:
        return "\nSo the final answer is: Paris City"
    if prompt.endswith("So the final answer is:"):
        return " Paris City"
    if prompt.rstrip().endswith("A:"):
        return " John Smith directed Gamma Film. So the answer is: Paris City."
    return " John Smith was born in Paris City. So the answer is: Paris City."


def e1_responder(req, i, rng):
    if "prompt" in req:
        return completion_responder(req, i, rng)
    sys_msg = req["messages"][0]["content"]
    if sys_msg.startswith("You are a question decomposition agent"):
        return '["Who directed Gamma Film?", "Where was #1 born?"]'
    if sys_msg.startswith("You are a planning agent"):
        return ('```python\ndocs1 = retrieve("Who directed Gamma Film?")\nd = answer("Who directed Gamma Film?", docs1)\n'
                'docs2 = retrieve(f"Where was {d} born?")\nb = answer(f"Where was {d} born?", docs2)\n'
                'final_answer = answer(f"Given: {d} was born in {b}. Answer the question: Where?")\n```')
    if sys_msg.startswith("You are given a question and retrieved documents"):
        return "<redacted_thinking>Doc [1].</redacted_thinking>\n<answer> unknown </answer>" if rng.random() < 0.3 else \
               "<redacted_thinking>Doc [1].</redacted_thinking>\n<answer> John Smith </answer>"
    if sys_msg.startswith("There are NO retrieved documents"):
        return "<redacted_thinking>ok</redacted_thinking>\n<answer> Paris City </answer>"
    if sys_msg.startswith("**Task Instruction:**"):
        return "**Final Information**\n\nGamma Film was directed by John Smith."
    if sys_msg.startswith("You are a reasoning assistant") or req["messages"][0]["role"] == "user" and "begin_search_query" in sys_msg:
        last = req["messages"][-1]
        if last["role"] == "assistant" and "begin_search_result" in last["content"]:
            return "So the answer is \\boxed{Paris City}."
        return "I need to search. <|begin_search_query|>Gamma Film director<|end_search_query|>"

    user = req["messages"][-1]["content"]
    temp = req.get("temperature", 0)
    if sys_msg.startswith("Decompose"):
        q = [m["content"] for m in req["messages"] if m["role"] == "user"][-1]
        if "Question:" not in q:   # retry turn: find the original question
            q = [m["content"] for m in req["messages"] if m["role"] == "user" and "Question:" in m["content"]][-1]
        q = q.split("Question:", 1)[1].strip()
        if "BADPLAN" in q:
            return "I cannot do that"
        if " or " in q:
            a, b = re.findall(r"(Alpha \w+|Beta \w+)", q)[:2] or ["Alpha One", "Beta Two"]
            return json.dumps({"steps": [
                {"id": 1, "kind": "TXT", "question": f"When was {a} founded?", "depends_on": []},
                {"id": 2, "kind": "TXT", "question": f"When was {b} founded?", "depends_on": []},
                {"id": 3, "kind": "CMP", "question": f"Which was founded earlier, {a} (#1) or {b} (#2)?", "depends_on": [1, 2]}]})
        if q.startswith("Are"):
            return json.dumps({"steps": [
                {"id": 1, "kind": "TXT", "question": "Who directed Gamma Film?", "depends_on": []},
                {"id": 2, "kind": "TXT", "question": "Who directed Delta Film?", "depends_on": []},
                {"id": 3, "kind": "LOG", "question": "Are #1 and #2 the same person?", "depends_on": [1, 2]}]})
        return "```json\n" + json.dumps({"steps": [
            {"id": 1, "kind": "TXT", "question": "Who directed Gamma Film?", "depends_on": []},
            {"id": "2", "kind": "txt", "question": "Where was #1 born?", "depends_on": ["#1"]}]}) + "\n```"
    if sys_msg.startswith("Answer the sub-question"):
        # pick the first capitalised word pair after the first passage title, vary with sampling
        passages = re.findall(r"^\[(\d+)\] ([^:\n]+): (.*)$", user, flags=re.M)
        if not passages:
            return "Answer: NOT FOUND\nPassages: none"
        j = 0 if temp == 0 else rng.randint(0, min(2, len(passages) - 1))
        pid, title, text = passages[j]
        m = re.search(r"(\d{4}|[A-Z][a-z]+ [A-Z][a-z]+)", text)
        ans = m.group(1) if m else "NOT FOUND"
        return f"Answer: {ans}\nPassages: {pid}"
    if sys_msg.startswith("You compute one step"):
        if temp > 0 and rng.random() < 0.3:
            return "Expression: undefined_name + 1"
        nums = re.findall(r"\b(1[89]\d\d|20\d\d)\b", user.split("Step:")[0])
        names = re.findall(r"(Alpha \w+|Beta \w+)", user.split("Step:")[1])
        if len(nums) >= 2 and len(names) >= 2:
            return f'Expression: "{names[0]}" if {nums[0]} < {nums[1]} else "{names[1]}"'
        return "Expression: date(1999, 2, 30) < 3"
    if sys_msg.startswith("You complete one step") and "Reason briefly" in sys_msg:
        return "Reasoning: The two results differ.\nAnswer: no"
    if sys_msg.startswith("You complete one step"):
        return "Answer: " + ("yes" if temp == 0 else rng.choice(["yes", "no"]))
    if sys_msg.startswith("Write a keyword search query"):
        return user.split("Query:", 1)[-1].strip() + " history"
    if sys_msg.startswith("Two candidate answers to the same sub-question"):
        return "Supported: B"
    if sys_msg.startswith("Several candidate answers"):
        return "Supported: B"
    if sys_msg.startswith("Two candidate answers to one step"):
        return "Correct: A"
    return "Answer: unknown"


class H(BaseHTTPRequestHandler):
    responder = None

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps(STATS).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with LOCK:
            STATS["requests"] += 1
        n = req.get("n", 1)
        rng = random.Random(json.dumps(req, sort_keys=True))
        choices, comp = [], 0
        for i in range(n):
            text = H.responder(req, i, rng)
            toks = tokens_of(text)
            comp += len(toks)
            c = {"index": i, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
            if req.get("logprobs"):
                c["logprobs"] = {"content": [{"token": t, "logprob": -rng.random() * (0.05 if req.get("temperature", 0) == 0 else 1.5),
                                              "bytes": list(t.encode()), "top_logprobs": []} for t in toks]}
            choices.append(c)
        prompt = len(req["prompt"].split()) if "prompt" in req else sum(len(m["content"].split()) for m in req["messages"])
        body = json.dumps({"choices": choices, "usage": {"prompt_tokens": prompt, "completion_tokens": comp}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = int(sys.argv[1])
    H.responder = {"echo": echo_responder, "e1": e1_responder}[sys.argv[2] if len(sys.argv) > 2 else "echo"]
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
