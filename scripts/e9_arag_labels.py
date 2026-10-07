"""Adaptive-RAG baseline, step 1: silver complexity labels for the calibration questions (and the same rule on test).

Why: Adaptive-RAG (Jeong et al., NAACL 2024; official code github.com/starsuzi/Adaptive-RAG, commit 0c88670) routes each
question before execution with a T5-large classifier trained on automatically collected labels. Its labelling rule
(paper §3.3; code classifier/preprocess/preprocess_utils.py:label_complexity, correctness = normalised exact match):
the simplest strategy that answers correctly wins, A = no retrieval, B = single-step retrieval, C = multi-step (IRCoT);
questions that no strategy answers get the dataset's inductive-bias label (B for single-hop, C for multi-hop datasets).
Our three datasets are all multi-hop, so the inductive-bias label is always C. Mapped to our cached runs:
  A  CoT (no retrieval)        results/e2/<tag>/<split>/<ds>/simple_s17.jsonl  field "cot"
  B  Standard RAG (one-shot)   same file, field "rag"
  C  IRCoT (official setting)  results/e2/<tag>/<split>/<ds>/ircot_k<K>.jsonl
Second label set, for the ablation "pre-execution routing over our own two strategies": CHEAP if Plan-Only is correct,
else EXPENSIVE if IRCoT is correct, else CHEAP (when neither answers, the cheap strategy is the right call).
  Plan-Only  cal: results/e3/<tag>/cal/<ds>/questions.jsonl field "base"; test: results/e3/<tag>/test/<ds>/runs/plan_only_s17.jsonl
Deviation from the official code (also in results/e9/<tag>/arag/summary.md): the code drops the
questions no strategy answers and instead adds 400 separate training-split questions per dataset labelled by inductive
bias; we follow the paper's text and give the unanswered calibration questions the inductive-bias label (C), because with
only multi-hop datasets the code's extra 1,200 all-C questions would push the classifier to always answer C.
Input:  results/e0/ids/<ds>_{cal,test}.txt, the files above (cal outputs copied from the servers into results/e2/<tag>/cal/)
Output: results/e9/<tag>/arag/labels_abc.jsonl, labels_plan_ircot.jsonl   one row per calibration question:
          {qid, ds, question, label, source}   source = "silver" (some strategy correct) or "inductive"/"default"
        results/e9/<tag>/arag/test_questions.jsonl                         {qid, ds, question} for the classifier to label
        (test silver labels are recomputed with the same functions by scripts/e9_arag_eval.py; never shown to training)
Usage:  python3 scripts/e9_arag_labels.py --tag llama31-8b [--root <dir>]      standard library only; --root for toy tests
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
# IRCoT's paragraph count K per dataset, chosen on the tuning split in e2 (official IRCoT setting otherwise)
IRCOT_K = {"llama31-8b": {"hotpotqa": 6, "2wikimultihopqa": 8, "musique": 6},
           "qwen3-8b": {"hotpotqa": 6, "2wikimultihopqa": 6, "musique": 4}}
STRATS = ("cot", "rag", "ircot", "plan")
LABELS = {"abc": ("A", "B", "C"), "plan_ircot": ("CHEAP", "EXPENSIVE")}
ROUTE = {"abc": {"A": "cot", "B": "rag", "C": "ircot"}, "plan_ircot": {"CHEAP": "plan", "EXPENSIVE": "ircot"}}
INDUCTIVE = {ds: "C" for ds in DATASETS}   # all three are multi-hop datasets (single-hop ones would get "B")


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def _cell(r):
    tok = r["tok"]
    return {"em": float(r["em"]), "f1": float(r["f1"]), "tok": sum(tok.values()) if isinstance(tok, dict) else tok,
            "pred": r.get("pred")}


def _by_qid(rows, path, ids):
    d = {r["qid"]: r for r in rows}
    missing = [q for q in ids if q not in d]
    if missing:
        raise SystemExit(f"{path}: {len(missing)}/{len(ids)} questions missing (e.g. {missing[0]}); is the run finished?")
    return d


def load_split(tag, split, root=ROOT):
    """{ds: {qid: {"question", "cot", "rag", "ircot", "plan": {em, f1, tok, pred}}}} in the order of results/e0/ids."""
    assert split in ("cal", "test"), split
    out = {}
    for ds in DATASETS:
        ids = [l.strip() for l in open(root / "results" / "e0" / "ids" / f"{ds}_{split}.txt") if l.strip()]
        e2 = root / "results" / "e2" / tag / split / ds
        e3 = root / "results" / "e3" / tag / split / ds
        simple_p, ircot_p, q_p = e2 / "simple_s17.jsonl", e2 / f"ircot_k{IRCOT_K[tag][ds]}.jsonl", e3 / "questions.jsonl"
        simple = _by_qid(read_jsonl(simple_p), simple_p, ids)
        ircot = _by_qid(read_jsonl(ircot_p), ircot_p, ids)
        qs = _by_qid(read_jsonl(q_p), q_p, ids)
        if split == "cal":
            plan = {q: qs[q]["base"] for q in ids}
        else:
            plan_p = e3 / "runs" / "plan_only_s17.jsonl"
            plan = _by_qid(read_jsonl(plan_p), plan_p, ids)
        out[ds] = {q: {"question": qs[q]["question"], "cot": _cell(simple[q]["cot"]), "rag": _cell(simple[q]["rag"]),
                       "ircot": _cell(ircot[q]), "plan": _cell(plan[q])} for q in ids}
    return out


def label_abc(r, ds):
    """Adaptive-RAG's rule: simplest correct strategy (A < B < C); none correct -> the dataset's inductive-bias label."""
    for lab, s in (("A", "cot"), ("B", "rag"), ("C", "ircot")):
        if r[s]["em"] == 1.0:
            return lab, "silver"
    return INDUCTIVE[ds], "inductive"


def label_plan_ircot(r, ds):
    """Ablation: CHEAP if Plan-Only is correct, else EXPENSIVE if IRCoT is correct, else CHEAP."""
    if r["plan"]["em"] == 1.0:
        return "CHEAP", "silver"
    if r["ircot"]["em"] == 1.0:
        return "EXPENSIVE", "silver"
    return "CHEAP", "default"


LABEL_FN = {"abc": label_abc, "plan_ircot": label_plan_ircot}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, choices=sorted(IRCOT_K))
    ap.add_argument("--root", default=str(ROOT), help="project root (a scratch copy for toy tests)")
    args = ap.parse_args()
    root = Path(args.root)
    out_dir = root / "results" / "e9" / args.tag / "arag"
    out_dir.mkdir(parents=True, exist_ok=True)
    cal = load_split(args.tag, "cal", root)
    for name, fn in LABEL_FN.items():
        rows = [{"qid": q, "ds": ds, "question": r["question"], **dict(zip(("label", "source"), fn(r, ds)))}
                for ds in DATASETS for q, r in cal[ds].items()]
        p = out_dir / f"labels_{name}.jsonl"
        p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
        print(f"wrote {p} ({len(rows)} rows)")
        for ds in DATASETS:
            c = Counter((x["label"], x["source"]) for x in rows if x["ds"] == ds)
            print(f"  {ds:16s} " + "  ".join(f"{l}/{s}={n}" for (l, s), n in sorted(c.items())))
    # test questions only (no labels), for the classifier to predict on
    test_qs = {ds: [l.strip() for l in open(root / "results" / "e0" / "ids" / f"{ds}_test.txt") if l.strip()]
               for ds in DATASETS}
    rows = []
    for ds in DATASETS:
        qp = root / "results" / "e3" / args.tag / "test" / ds / "questions.jsonl"
        qs = _by_qid(read_jsonl(qp), qp, test_qs[ds])
        rows += [{"qid": q, "ds": ds, "question": qs[q]["question"]} for q in test_qs[ds]]
    p = out_dir / "test_questions.jsonl"
    p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    print(f"wrote {p} ({len(rows)} rows)")
    overlap = {x["qid"] for x in rows} & {q for ds in DATASETS for q in cal[ds]}
    if overlap:
        raise SystemExit(f"{len(overlap)} calibration questions also in test")


if __name__ == "__main__":
    main()
