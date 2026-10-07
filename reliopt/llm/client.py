"""Client for local vLLM servers (OpenAI-compatible chat API) with a disk cache and token accounting.

Only local endpoints are accepted (localhost / 127.0.0.1), so no request can reach a paid API by accident.
Every call returns its token usage; a cache hit returns the usage of the original call, so a strategy's reported cost
counts every call it would have issued even when another strategy already paid for it.
Cost convention (same for every method): prompt tokens once per request + completion tokens of all returned samples.
Identical requests issued concurrently are coalesced (one goes to the server, the others wait for its answer), so two threads
can never end up with different sampled outputs for the same cache key.
"""
import hashlib
import json
import sqlite3
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from urllib.parse import urlparse

import requests

LOCAL_HOSTS = {"localhost", "127.0.0.1"}

# Models used in this paper: tag -> served name in run/e0_serve.sh and chat-template switches (Qwen3 thinking off).
MODELS = {
    "llama31-8b": {"served": "llama31-8b", "chat_template_kwargs": None},
    "qwen3-8b": {"served": "qwen3-8b", "chat_template_kwargs": {"enable_thinking": False}},
    "qwen3-8b-think": {"served": "qwen3-8b", "chat_template_kwargs": {"enable_thinking": True}},
}


class Cache:
    """sqlite key -> response json; one connection per thread, WAL so readers and writers do not block each other."""

    def __init__(self, path):
        self.path = str(path)
        self._local = threading.local()
        con = self._con()
        con.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        con.commit()

    def _con(self):
        if not hasattr(self._local, "con"):
            con = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
            con.execute("PRAGMA journal_mode=WAL")
            self._local.con = con
        return self._local.con

    def get(self, k):
        row = self._con().execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k, v):
        """Retries when the database is locked: many processes (sharded runs) write to one cache file."""
        con = self._con()
        row = (k, json.dumps(v, ensure_ascii=False))
        for attempt in range(20):
            try:
                con.execute("INSERT OR REPLACE INTO c VALUES (?, ?)", row)
                con.commit()
                return
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) or attempt == 19:
                    raise
                time.sleep(min(30.0, 0.5 * 2 ** attempt))


class LLM:
    """chat(messages, ...) -> {"texts": [...], "logprobs": [[...]] or None, "tokens": [[...]] or None, "finish": [...],
    "usage": {"prompt", "completion"}, "cached"}. With logprobs=True, logprobs[i] and tokens[i] list the generated tokens of
    sample i, so len(logprobs[i]) is that sample's completion length.

    `tag` names the model in MODELS and is part of every cache key; each call goes to the least-loaded endpoint.
    """

    def __init__(self, tag, endpoints, cache_path=None, timeout=600, retries=4):
        assert tag in MODELS, tag
        for e in endpoints:
            host = urlparse(e).hostname
            if host not in LOCAL_HOSTS:
                raise ValueError(f"refusing non-local endpoint {e}: this project never calls remote or paid APIs")
        self.tag, self.endpoints, self.timeout, self.retries = tag, list(endpoints), timeout, retries
        self.served_name = MODELS[tag]["served"]
        self.template_kwargs = MODELS[tag]["chat_template_kwargs"]
        self.cache = Cache(cache_path) if cache_path else None
        self._rr = 0
        self._lock = threading.Lock()
        self._local = threading.local()
        self._inflight = {}
        self.stats = {"calls": 0, "cache_hits": 0, "prompt": 0, "completion": 0}

    def _session(self):
        if not hasattr(self._local, "s"):
            s = requests.Session()
            s.trust_env = False   # ignore http(s)_proxy: the servers are local
            self._local.s = s
        return self._local.s

    def _next_endpoint(self, exclude=None):
        """Least-loaded routing: the endpoint with the fewest requests in flight from this process, per unit of weight (an
        endpoint listed k times gets weight k); ties go round-robin. Pure round-robin let a slow server collect most of the
        waiting threads while the others idled. Endpoints never enter the cache key."""
        with self._lock:
            if not hasattr(self, "_busy"):
                self._busy = {e: 0 for e in self.endpoints}
                self._weight = {e: self.endpoints.count(e) for e in self.endpoints}
                self._uniq = list(dict.fromkeys(self.endpoints))
                self._down = {}   # endpoint -> time until which it is skipped after a failed request
            now = time.time()
            up = [e for e in self._uniq if self._down.get(e, 0) <= now] or self._uniq
            cands = [e for e in up if e != exclude] or up
            k = self._rr % len(cands)
            self._rr += 1
            e = min(cands[k:] + cands[:k], key=lambda x: self._busy[x] / self._weight[x])
            self._busy[e] += 1
        return e

    def _release(self, e, failed=False):
        with self._lock:
            self._busy[e] -= 1
            if failed:   # a refused or failing server would otherwise look idle and draw every call
                self._down[e] = time.time() + 30

    def chat(self, messages, temperature=0.0, max_tokens=256, n=1, seed=None, logprobs=False, stop=None, top_p=1.0,
             extra=None):
        """Chat completion. `extra` adds vLLM request fields (e.g. top_k, repetition_penalty, continue_final_message)."""
        req = {"model": self.served_name, "messages": messages, "temperature": temperature, "top_p": top_p,
               "max_tokens": max_tokens, "n": n}
        if seed is not None:
            req["seed"] = seed
        if stop:
            req["stop"] = stop
        if logprobs:
            req["logprobs"] = True
            req["top_logprobs"] = 0
        if self.template_kwargs is not None:
            req["chat_template_kwargs"] = self.template_kwargs
        if extra:
            req.update(extra)
        return self._cached(req, {"tag": self.tag, **req}, "/v1/chat/completions")

    def complete(self, prompt, temperature=0.0, max_tokens=256, n=1, seed=None, stop=None, top_p=1.0, extra=None):
        """Plain-text completion (no chat template), for few-shot prompts written for completion models (IRCoT,
        Self-Ask). Returns the same dict as chat() without logprobs."""
        req = {"model": self.served_name, "prompt": prompt, "temperature": temperature, "top_p": top_p,
               "max_tokens": max_tokens, "n": n}
        if seed is not None:
            req["seed"] = seed
        if stop:
            req["stop"] = stop
        if extra:
            req.update(extra)
        return self._cached(req, {"tag": self.tag, "endpoint": "completions", **req}, "/v1/completions")

    def _cached(self, req, key_obj, path):
        key = hashlib.sha256(json.dumps(key_obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with self._lock:
            fut = self._inflight.get(key)
            owner = fut is None
            if owner:
                fut = self._inflight[key] = Future()
        if not owner:   # the same request is already in flight in another thread: share its answer
            out = fut.result()
            self._count(out, cached=True)
            return {**out, "cached": True}
        try:
            hit = self.cache.get(key) if self.cache is not None else None
            if hit is not None:
                fut.set_result(hit)
                self._count(hit, cached=True)
                return {**hit, "cached": True}
            out = self._post(req, path)
            if self.cache is not None:
                self.cache.put(key, out)
            fut.set_result(out)
        except BaseException as e:
            fut.set_exception(e)
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)
        self._count(out, cached=False)
        return {**out, "cached": False}

    def _post(self, req, path):
        err = last = None
        for attempt in range(self.retries):
            ep = last = self._next_endpoint(exclude=last)
            url = ep.rstrip("/") + path
            try:
                try:
                    r = self._session().post(url, json=req, timeout=self.timeout)
                except Exception:
                    self._release(ep, failed=True)
                    raise
                self._release(ep, failed=r.status_code >= 500)
                r.raise_for_status()
                js = r.json()
                choices = sorted(js["choices"], key=lambda c: c["index"])
                texts = [(c["message"]["content"] if "message" in c else c.get("text")) or "" for c in choices]
                lps = toks = None
                if req.get("logprobs"):
                    contents = [((c.get("logprobs") or {}).get("content") or []) for c in choices]
                    lps = [[t["logprob"] for t in cs] for cs in contents]
                    toks = [[t["token"] for t in cs] for cs in contents]
                u = js.get("usage") or {}
                return {"texts": texts, "logprobs": lps, "tokens": toks, "finish": [c.get("finish_reason") for c in choices],
                        "usage": {"prompt": u.get("prompt_tokens", 0), "completion": u.get("completion_tokens", 0)}}
            except Exception as e:   # transient server errors: back off and try the next endpoint
                err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM request failed after {self.retries} attempts: {err}")

    def _count(self, out, cached):
        with self._lock:
            self.stats["calls"] += 1
            self.stats["cache_hits"] += int(cached)
            self.stats["prompt"] += out["usage"]["prompt"]
            self.stats["completion"] += out["usage"]["completion"]


def run_parallel(fn, items, workers=64):
    """Apply fn to items with a thread pool, preserving order (vLLM batches the concurrent requests)."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))
