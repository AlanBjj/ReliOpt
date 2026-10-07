"""Answer metrics for multi-hop QA: SQuAD-style normalisation, EM, F1 and answer containment (Acc).

EM/F1 follow the SQuAD v1.1 script that IRCoT's evaluation builds on: lower-case, strip punctuation and the articles a/an/the,
collapse whitespace, then take the maximum over all gold answers. Acc is 1 when a normalised gold answer occurs inside the
normalised prediction (the "Acc" column of Adaptive-RAG).
"""
import collections
import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def normalize_answer(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _f1(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def score(pred, golds):
    """Return {"em", "f1", "acc"} of one prediction against a list of gold answers (max over golds)."""
    golds = [g for g in golds if str(g).strip()] or [""]
    pred = "" if pred is None else str(pred)
    np_ = normalize_answer(pred)
    em = max(float(np_ == normalize_answer(g)) for g in golds)
    f1 = max(_f1(pred, g) for g in golds)
    acc = max(float(bool(normalize_answer(g)) and normalize_answer(g) in np_) for g in golds)
    return {"em": em, "f1": f1, "acc": acc}


def aggregate(rows):
    """Mean of em/f1/acc over a list of score dicts, in percent."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "em": 0.0, "f1": 0.0, "acc": 0.0}
    return {"n": n, **{k: 100.0 * sum(r[k] for r in rows) / n for k in ("em", "f1", "acc")}}
