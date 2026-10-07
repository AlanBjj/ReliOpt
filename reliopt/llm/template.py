"""Render a chat as the model's own prompt text (tokenizer.apply_chat_template), for methods whose official code appends
generated text to a templated prompt and continues it with plain completion (Search-o1). Tokenizers load from the Hugging Face
id below or from MODEL_DIR, a local copy of the same weights (as in run/e0_serve.sh); RELIOPT_FAKE_TEMPLATE=1 uses a plain-text stand-in for local tests."""
import json
import os
import threading

MODEL_PATHS = {"llama31-8b": "meta-llama/Llama-3.1-8B-Instruct",
               "qwen3-8b": "Qwen/Qwen3-8B", "qwen3-8b-think": "Qwen/Qwen3-8B"}
THINKING = {"qwen3-8b": False, "qwen3-8b-think": True}
FAMILY = {"llama31-8b": "llama", "qwen3-8b": "qwen3", "qwen3-8b-think": "qwen3"}   # config.json model_type


_LOCK, _TOK = threading.Lock(), {}


def _tokenizer(tag):
    """The tag's tokenizer. MODEL_DIR must hold the same model family: a stale MODEL_DIR inherited from another
    session would render one model's prompts with the other's chat template."""
    path = os.environ.get("MODEL_DIR") or MODEL_PATHS[tag]
    # one thread loads it: transformers imports lazily, and concurrent first imports see a half-initialised module
    with _LOCK:
        if path not in _TOK:
            cfg = os.path.join(path, "config.json")   # a Hugging Face id has no local config.json to check
            mt = json.loads(open(cfg).read()).get("model_type") if os.path.isfile(cfg) else FAMILY[tag]
            if mt != FAMILY[tag]:
                raise ValueError(f"{path} holds a {mt} model, not {FAMILY[tag]} ({tag}): fix MODEL_DIR")
            from transformers import AutoTokenizer
            _TOK[path] = AutoTokenizer.from_pretrained(path)
        return _TOK[path]


def render(tag, messages):
    """Prompt text for `messages` with the assistant turn opened (add_generation_prompt=True)."""
    if os.environ.get("RELIOPT_FAKE_TEMPLATE"):
        return "".join(f"<{m['role']}>\n{m['content']}\n" for m in messages) + "<assistant>\n"
    kw = {"enable_thinking": THINKING[tag]} if tag in THINKING else {}
    return _tokenizer(tag).apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
