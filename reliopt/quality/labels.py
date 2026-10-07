"""Step labels from the intermediate annotations; used for calibration and offline diagnostics only.

Gold intermediate answers: MuSiQue hop answers (reasoning_steps "... >>>> answer"); 2WikiMultiHopQA evidence-triple objects;
HotpotQA supporting-paragraph titles (the bridge entity is one of them) plus, for attribute look-ups, the supporting text.

A step is labelled given the labels of the steps it uses:
  - any parent not "correct"                      -> "undet" (its local error cannot be judged against the gold path)
  - the sink                                      -> "correct" iff the final answer is an exact match (official EM)
  - TXT returning NOT FOUND                       -> "error"
  - TXT matching a gold intermediate / final answer (normalised exact, date-compatible, or token containment) that its
    own sub-question does not already name, or for HotpotQA appearing in a supporting paragraph -> "correct"
  - TXT otherwise                                 -> "error" if the step is aligned with the gold path (its sub-question
    covers a gold hop or names a gold entity), else "undet"
  - CMP / LOG (not sink) matching a gold answer   -> "correct", otherwise "undet"
"""
import re

from ..eval.metrics import normalize_answer, score
from ..exec.steps import is_abstain

_MONTHS = {m: i for i, m in enumerate(("january february march april may june july august september october november "
                                       "december").split(), 1)}
_STOP = set("a an the of in on at to for by with from and or is are was were be been who whom whose what which when where "
            "how did does do this that these those it its as into than then there their they he she his her".split())


def _date_key(s):
    t = normalize_answer(s)
    m = re.fullmatch(r"(\d{1,2}) ([a-z]+) (\d{3,4})", t) or None
    if m and m.group(2) in _MONTHS:
        return int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1))
    m = re.fullmatch(r"([a-z]+) (\d{1,2}) (\d{3,4})", t)
    if m and m.group(1) in _MONTHS:
        return int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2))
    m = re.fullmatch(r"([a-z]+) (\d{3,4})", t)
    if m and m.group(1) in _MONTHS:
        return int(m.group(2)), _MONTHS[m.group(1)], None
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", t)   # 1887-09-04 after punctuation removal
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.fullmatch(r"(\d{3,4})", t)
    if m:
        return int(m.group(1)), None, None
    return None


# Country names and nationality adjectives are interchangeable answers to "country of citizenship / origin" hops.
_COUNTRY_GROUPS = [
    "united states|united states of america|usa|us|u s|america|american", "united kingdom|uk|britain|great britain|british",
    "england|english", "scotland|scottish", "wales|welsh", "ireland|irish", "netherlands|holland|dutch",
    "germany|german|west germany|east germany", "france|french", "italy|italian", "spain|spanish", "portugal|portuguese",
    "egypt|egyptian", "australia|australian", "canada|canadian", "india|indian", "japan|japanese", "china|chinese",
    "russia|russian|russian federation", "soviet union|ussr|soviet", "mexico|mexican", "sweden|swedish", "norway|norwegian",
    "denmark|danish", "finland|finnish", "iceland|icelandic", "poland|polish", "czech republic|czechia|czech",
    "czechoslovakia|czechoslovak", "slovakia|slovak", "hungary|hungarian", "austria|austrian", "switzerland|swiss",
    "belgium|belgian", "brazil|brazilian", "argentina|argentine|argentinian", "chile|chilean", "colombia|colombian",
    "peru|peruvian", "cuba|cuban", "greece|greek", "turkey|turkish", "iran|iranian|persian", "israel|israeli",
    "south korea|korea|korean|south korean", "north korea|north korean", "philippines|filipino|philippine",
    "indonesia|indonesian", "thailand|thai", "vietnam|vietnamese", "malaysia|malaysian", "pakistan|pakistani",
    "bangladesh|bangladeshi", "sri lanka|sri lankan", "nigeria|nigerian", "kenya|kenyan", "south africa|south african",
    "new zealand|new zealander", "romania|romanian", "bulgaria|bulgarian", "ukraine|ukrainian", "croatia|croatian",
    "serbia|serbian", "slovenia|slovenian", "bosnia and herzegovina|bosnian", "yugoslavia|yugoslav|yugoslavian",
    "estonia|estonian", "latvia|latvian", "lithuania|lithuanian", "georgia|georgian", "armenia|armenian",
    "lebanon|lebanese", "syria|syrian", "iraq|iraqi", "saudi arabia|saudi", "morocco|moroccan", "algeria|algerian",
    "tunisia|tunisian", "ethiopia|ethiopian", "ghana|ghanaian", "venezuela|venezuelan", "uruguay|uruguayan",
    "taiwan|taiwanese", "singapore|singaporean", "hong kong", "luxembourg|luxembourgish",
]
_COUNTRY = {name: g.split("|")[0] for g in _COUNTRY_GROUPS for name in g.split("|")}


def _country(s):
    return _COUNTRY.get(normalize_answer(s))


def _contains(a, b):
    n = len(b)
    return any(a[i:i + n] == b for i in range(len(a) - n + 1))


def match(pred, gold):
    p, g = normalize_answer(pred), normalize_answer(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    cp, cg = _country(pred), _country(gold)
    if cp and cg:
        return cp == cg
    dp, dg = _date_key(pred), _date_key(gold)
    if dp and dg:
        return all(x is None or y is None or x == y for x, y in zip(dp, dg))
    pt, gt = p.split(), g.split()
    if _contains(pt, gt) and len(pt) <= len(gt) + 3:
        return True
    return _contains(gt, pt) and len(pt) >= max(1, (len(gt) + 1) // 2)


def _strip_paren(s):
    return re.sub(r"\s*\([^)]*\)", "", s).strip()


def _stems(text):
    return {w[:5] for w in normalize_answer(text).split() if w not in _STOP and len(w) > 2}


def gold_view(q):
    """Gold intermediate answers, alignment hints and finals of one question."""
    ds = q["dataset"]
    view = {"ds": ds, "finals": [a for a in q["answers"] if str(a).strip()], "cands": [], "entities": [], "hops": [],
            "texts": []}
    if ds == "musique":
        view["cands"] = [a for _, a in q["steps"] if a]
        view["hops"] = [h for h, _ in q["steps"]]
    elif ds == "2wikimultihopqa":
        for text, obj in q["steps"]:
            if " | " in text and obj:
                subj, _ = text.rsplit(" | ", 1)
                view["cands"].append(obj)
                view["entities"] += [subj, obj]
    else:
        view["cands"] = list(q["support_titles"])
        view["entities"] = list(q["support_titles"])
        view["texts"] = list(q.get("support_texts", []))
    return view


def aligned(view, resolved):
    rq = f" {normalize_answer(resolved)} "
    for e in view["entities"]:
        ne = normalize_answer(_strip_paren(e))
        if ne and f" {ne} " in rq:
            return True
    sq = _stems(resolved)
    for h in view["hops"]:
        sh = _stems(h)
        if sh and len(sq & sh) / len(sh) >= 0.5:
            return True
    return False


def label_answer(view, kind, resolved, answer, parents_ok, is_sink):
    if not parents_ok:
        return "undet"
    if is_sink:
        return "correct" if score(answer, view["finals"])["em"] == 1.0 else "error"
    if kind == "TXT" and is_abstain(answer):
        return "error"
    # a gold string that the sub-question itself names cannot confirm the step (the model may just echo the entity)
    rq = f" {normalize_answer(resolved)} "
    golds = [g for g in view["cands"] + view["finals"] if f" {normalize_answer(g)} " not in rq]
    if any(match(answer, g) for g in golds):
        return "correct"
    if kind != "TXT":
        return "undet"
    na = normalize_answer(answer)
    if na and any(f" {na} " in f" {normalize_answer(t)} " for t in view["texts"]):
        return "correct"
    return "error" if aligned(view, resolved) else "undet"


def label_run(view, plan, run):
    """Labels of every step on one executed trajectory (a Pipeline.run(..., keep=True) result): {step id: label}."""
    sink = plan["steps"][-1]["id"]
    labels = {}
    for s in plan["steps"]:
        ctx = run["_ctx"][s["id"]]
        ans = next(r["ans"] for r in run["steps"] if r["id"] == s["id"])
        ok = all(labels[d] == "correct" for d in s["deps"])
        labels[s["id"]] = label_answer(view, s["kind"], ctx["resolved"], ans, ok, s["id"] == sink)
    return labels
