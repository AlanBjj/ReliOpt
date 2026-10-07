"""Free features phi_v of a step, available right after its default execution, no gold used."""
import re

from ..eval.metrics import normalize_answer
from ..exec.steps import is_abstain
from .labels import _date_key, _strip_paren

FEATURES = ("kind_TXT", "kind_CMP", "kind_LOG", "depth", "n_desc", "indeg", "n_steps", "is_sink",
            "form_yesno", "form_number", "form_date", "form_entity", "not_found", "lp_mean", "lp_min", "lp_missing",
            "truncated", "in_cited", "n_cited", "top_score", "score_gap", "title_match", "exec_ok", "fallback")


def answer_form(a):
    n = normalize_answer(a or "")
    if n in ("yes", "no"):
        return "yesno"
    if _date_key(a or "") and not re.fullmatch(r"\d{1,3}", n):
        return "date"
    if re.fullmatch(r"[\d.,]+( \w+)?", n):
        return "number"
    return "entity"


def step_features(struct, ctx, r0):
    """struct: planner.structure(plan steps)[step id]; ctx and r0 as produced by Pipeline.run(..., keep=True)."""
    kind = ctx["step"]["kind"]
    ans = r0["answer"]
    form = answer_form(ans)
    f = {f"kind_{k}": float(kind == k) for k in ("TXT", "CMP", "LOG")}
    f.update({k: float(struct[k]) for k in ("depth", "n_desc", "indeg", "n_steps", "is_sink")})
    f.update({f"form_{k}": float(form == k) for k in ("yesno", "number", "date", "entity")})
    f["not_found"] = float(is_abstain(ans))
    f["lp_missing"] = float(r0.get("lp_mean") is None)
    f["lp_mean"] = r0.get("lp_mean") or 0.0
    f["lp_min"] = r0.get("lp_min") or 0.0
    f["truncated"] = float(bool(r0.get("truncated")))
    hits = r0.get("hits") or []
    cited = [hits[i - 1] for i in r0.get("cited", []) if 1 <= i <= len(hits)]
    na = normalize_answer(ans)
    f["in_cited"] = float(bool(na) and any(f" {na} " in f" {normalize_answer(h['title'] + ' ' + h['text'])} "
                                           for h in cited))
    f["n_cited"] = float(len(cited))
    scores = [h["score"] for h in hits]
    f["top_score"] = scores[0] if scores else 0.0
    f["score_gap"] = scores[0] - scores[1] if len(scores) > 1 else (scores[0] if scores else 0.0)
    rq = f" {normalize_answer(ctx['resolved'])} "
    f["title_match"] = float(any(normalize_answer(_strip_paren(h["title"])) and
                                 f" {normalize_answer(_strip_paren(h['title']))} " in rq for h in cited))
    f["exec_ok"] = float(bool(r0.get("exec_ok"))) if kind == "CMP" else 1.0
    f["fallback"] = float(bool(r0.get("fallback")))   # CMP answered directly, or TXT answered closed-book
    return f
