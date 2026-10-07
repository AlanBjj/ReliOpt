"""Loaders for the three multi-hop QA datasets in IRCoT's processed format, plus the step annotations used for labels.

Layout under cache/ircot/ (built by scripts/e0_prepare_data.py):
  processed_data/<ds>/{train,dev,dev_subsampled,test_subsampled}.jsonl   IRCoT release (test: 500, dev: 100 per dataset)
  raw_data/<ds>/...                                                     official raw files (for corpora, 2Wiki types)
  corpus/<ds>.jsonl                                                     {"id", "title", "text"} per paragraph

A question is a dict:
  qid, dataset, question, answers (list of gold strings), qtype (bridge/comparison/... or "<k>hop"),
  steps (gold reasoning steps as (sub-question or evidence text, intermediate answer) pairs, may be empty),
  support_titles / support_texts (titles and text of supporting paragraphs).
Gold fields are for labels and diagnostics only, never model input.
"""
import json
import re
from pathlib import Path

from ..utils.env import ROOT

DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
IRCOT_DIR = ROOT / "cache" / "ircot"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _answers(inst):
    out = []
    for obj in inst.get("answers_objects", []):
        out += [s for s in obj.get("spans", []) if str(s).strip()]
        if str(obj.get("number", "")).strip():
            out.append(str(obj["number"]))
        d = obj.get("date") or {}
        if any(str(d.get(k, "")).strip() for k in ("day", "month", "year")):
            out.append(" ".join(str(d.get(k, "")) for k in ("day", "month", "year")).strip())
    return out


def _musique_steps(inst):
    steps = []
    for s in inst.get("reasoning_steps", []):
        q, _, a = s.partition(">>>>")
        steps.append((q.strip(), a.strip()))
    return steps


def _twowiki_steps(raw):
    # raw 2Wiki "evidences": [[subject, relation, object], ...]; the object is the intermediate answer of that hop
    return [(f"{s} | {r}", o) for s, r, o in raw.get("evidences", [])] if raw else []


def _hop_type(qid):
    m = re.match(r"^(\d)hop", qid)
    return f"{m.group(1)}hop" if m else "unknown"


def load_twowiki_raw(split):
    """Raw 2WikiMultiHopQA split as {qid: item}; carries 'type' and 'evidences' that IRCoT's processed files drop."""
    path = IRCOT_DIR / "raw_data" / "2wikimultihopqa" / f"{split}.json"
    if not path.exists():
        return {}
    return {it["_id"]: it for it in json.load(open(path))}


_QID = re.compile(r'"question_id":\s*"([^"]+)"')


def _iter_instances(path, want):
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            if want is not None:   # skip unwanted lines before parsing them (the train files are 0.2-0.8 GB)
                m = _QID.search(line)
                if m and m.group(1) not in want:
                    continue
            yield json.loads(line)


def load_split(ds, split, raw_index=None, qids=None):
    """Load one processed split (train / dev / dev_subsampled / test_subsampled) as a list of question dicts,
    optionally only the questions in `qids` (file order is kept)."""
    assert ds in DATASETS, ds
    path = IRCOT_DIR / "processed_data" / ds / f"{split}.jsonl"
    want = set(qids) if qids is not None else None
    rows = []
    for inst in _iter_instances(path, want):
        qid = inst["question_id"]
        if want is not None and qid not in want:
            continue
        q = {"qid": qid, "dataset": ds, "question": inst["question_text"], "answers": _answers(inst),
             "support_titles": [c["title"] for c in inst.get("contexts", []) if c.get("is_supporting")],
             "support_texts": [c.get("paragraph_text", "") for c in inst.get("contexts", []) if c.get("is_supporting")]}
        if ds == "hotpotqa":
            q["qtype"], q["steps"] = inst.get("type", "unknown"), []
        elif ds == "musique":
            q["qtype"], q["steps"] = _hop_type(qid), _musique_steps(inst)
        else:
            raw = (raw_index or {}).get(qid)
            q["qtype"] = raw.get("type", "unknown") if raw else "unknown"
            q["steps"] = _twowiki_steps(raw) if raw else [(s, "") for s in inst.get("reasoning_steps", [])]
        rows.append(q)
    return rows


def load_questions(ds, split, qids=None):
    """load_split with the raw 2Wiki index attached when needed (dev for dev/test subsets, train for train)."""
    raw = None
    if ds == "2wikimultihopqa":
        raw = load_twowiki_raw("train" if split == "train" else "dev")
        if qids is not None:
            raw = {k: raw[k] for k in set(qids) if k in raw}
    return load_split(ds, split, raw, qids)
