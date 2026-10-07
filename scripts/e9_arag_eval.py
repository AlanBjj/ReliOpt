"""Adaptive-RAG baseline, step 3: route every test question with the classifier's label to the cached strategy outputs.

Why: Adaptive-RAG decides before execution, so its answer and cost for a question are exactly those of the strategy its
classifier picks; the routed result can be read off the cached per-question runs without new LLM calls. Two routers:
  Adaptive-RAG            pred_abc:        A -> CoT (no retrieval), B -> Standard RAG (single step), C -> IRCoT
  pre-exec {Plan, IRCoT}  pred_plan_ircot: CHEAP -> our Plan-Only, EXPENSIVE -> IRCoT   (ablation: the same pre-execution
                          router over our own two strategies, against our reliability-based escalation after execution)
Tokens per question are the routed strategy's tokens only (the router runs before execution; the T5-large classifier
costs one encoder pass and one decoder step and no LLM tokens, as in the Adaptive-RAG paper). Also reported: the single
strategies, each router with oracle (= silver) labels, label distributions (calibration silver, test silver, test
predicted), and the classifier's accuracy against test silver labels computed with the same rule as the training labels
(all 500 questions per dataset, and only the questions some strategy answers, which is what the official code validates on).
Input:  results/e2/<tag>/test/<ds>/{simple_s17,ircot_k<K>}.jsonl, results/e3/<tag>/test/<ds>/{questions.jsonl,
        runs/plan_only_s17.jsonl}, results/e9/<tag>/arag/{labels,pred,train}_{abc,plan_ircot}.json(l)
Output: results/e9/<tag>/arag/summary.{json,md}   (routers whose pred file is missing are listed as pending)
Usage:  python3 scripts/e9_arag_eval.py --tag llama31-8b [--root <dir>]      standard library only
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e9_arag_labels import DATASETS, LABEL_FN, LABELS, ROOT, ROUTE, load_split, read_jsonl  # noqa: E402

NAMES = {"cot": "CoT (no retrieval)", "rag": "Standard RAG (single step)", "ircot": "IRCoT", "plan": "Plan-Only (ours)"}
ROUTERS = (("arag", "abc", "pred", "Adaptive-RAG (T5-large router)"),
           ("arag_oracle", "abc", "silver", "Adaptive-RAG w/ oracle labels"),
           ("pre_plan_ircot", "plan_ircot", "pred", "Pre-exec router {Plan-Only, IRCoT} (T5-large)"),
           ("pre_plan_ircot_oracle", "plan_ircot", "silver", "Pre-exec router {Plan-Only, IRCoT} w/ oracle labels"))
DEVIATIONS = [
    "Labels come from our backbones' own runs (Llama-3.1-8B-Instruct / Qwen3-8B) instead of FLAN-T5-XL/XXL or GPT-3.5.",
    "Strategies are ours, on the same corpora and BM25: A = zero-shot CoT (official: IRCoT's no-retrieval prompt), "
    "B = Standard RAG with top-5 BM25 passages (official one-step: 15 paragraphs), C = IRCoT with the K tuned in e2.",
    "Classifier trained on our 3 x 500 calibration questions (training split) of the three multi-hop datasets only; "
    "the official classifier also uses NQ, TriviaQA and SQuAD and 500 dev questions per dataset.",
    "Unanswered calibration questions get the inductive-bias label C (paper text); the official code instead drops them "
    "and adds 400 other training-split questions per dataset with the inductive-bias label.",
    "Fixed 25 epochs (official: sweep 15-35 epochs, pick by accuracy on the test set's silver labels; their released "
    "post-processing uses epoch 25); fixed seed 17; trained in fp32 with a plain PyTorch loop on one GPU (same maths).",
    "Correctness for labels = IRCoT EM (normalised exact match against any gold answer), as in the official code.",
]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def cell_stats(rows):
    return {"em": 100 * mean([r["em"] for r in rows]), "f1": 100 * mean([r["f1"] for r in rows]),
            "tok": mean([r["tok"] for r in rows]), "n": len(rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default=str(ROOT), help="project root (a scratch copy for toy tests)")
    args = ap.parse_args()
    root = Path(args.root)
    adir = root / "results" / "e9" / args.tag / "arag"
    test = load_split(args.tag, "test", root)
    silver = {name: {ds: {q: fn(r, ds) for q, r in test[ds].items()} for ds in DATASETS} for name, fn in LABEL_FN.items()}
    preds, train_logs, cal_labels = {}, {}, {}
    for name in LABEL_FN:
        p = adir / f"pred_{name}.jsonl"
        if p.exists():
            rows = read_jsonl(p)
            preds[name] = {r["qid"]: r["pred"] for r in rows}
            want = {q for ds in DATASETS for q in test[ds]}
            if set(preds[name]) != want:
                raise SystemExit(f"{p}: {len(preds[name])} predictions, expected the {len(want)} test questions")
        lp = adir / f"train_{name}.json"
        if lp.exists():
            train_logs[name] = json.loads(lp.read_text())
        cp = adir / f"labels_{name}.jsonl"
        if cp.exists():
            cal_labels[name] = read_jsonl(cp)

    out = {"tag": args.tag, "methods": {}, "labels": {}, "classifier": {}, "train": train_logs, "pending": [],
           "deviations": DEVIATIONS}
    rag_tok = {ds: mean([r["rag"]["tok"] for r in test[ds].values()]) for ds in DATASETS}

    def add(key, name, per_ds):
        m = {"name": name}
        for ds in DATASETS:
            s = cell_stats(per_ds[ds])
            s["cost_x"] = s["tok"] / rag_tok[ds]
            m[ds] = s
        m["avg"] = {k: mean([m[ds][k] for ds in DATASETS]) for k in ("em", "f1", "tok", "cost_x")}
        out["methods"][key] = m

    for s, name in NAMES.items():
        add(s, name, {ds: [r[s] for r in test[ds].values()] for ds in DATASETS})
    for key, lset, src, name in ROUTERS:
        if src == "pred" and lset not in preds:
            out["pending"].append(f"{name}: results/e9/{args.tag}/arag/pred_{lset}.jsonl missing")
            continue
        lab = {ds: {q: (preds[lset][q] if src == "pred" else silver[lset][ds][q][0]) for q in test[ds]} for ds in DATASETS}
        add(key, name, {ds: [test[ds][q][ROUTE[lset][lab[ds][q]]] for q in test[ds]] for ds in DATASETS})
        out["methods"][key]["route_frac"] = {ds: {l: Counter(lab[ds].values())[l] / len(lab[ds]) for l in LABELS[lset]}
                                             for ds in DATASETS}

    for lset in LABEL_FN:
        d = {}
        if lset in cal_labels:
            d["cal_silver"] = {ds: dict(Counter(f"{r['label']}/{r['source']}" for r in cal_labels[lset] if r["ds"] == ds))
                               for ds in DATASETS}
        d["test_silver"] = {ds: dict(Counter(f"{l}/{s}" for l, s in silver[lset][ds].values())) for ds in DATASETS}
        if lset in preds:
            d["test_pred"] = {ds: dict(Counter(preds[lset][q] for q in test[ds])) for ds in DATASETS}
            acc = {}
            for ds in DATASETS + ("all",):
                qs = [(ds_, q) for ds_ in (DATASETS if ds == "all" else (ds,)) for q in test[ds_]]
                ok = [preds[lset][q] == silver[lset][ds_][q][0] for ds_, q in qs]
                okv = [preds[lset][q] == silver[lset][ds_][q][0] for ds_, q in qs if silver[lset][ds_][q][1] == "silver"]
                conf = Counter((silver[lset][ds_][q][0], preds[lset][q]) for ds_, q in qs)
                acc[ds] = {"acc_all": 100 * mean(ok), "n_all": len(ok), "acc_silver_only": 100 * mean(okv),
                           "n_silver_only": len(okv),
                           "confusion": {g: {p: conf[(g, p)] for p in LABELS[lset]} for g in LABELS[lset]}}
            out["classifier"][lset] = acc
        out["labels"][lset] = d

    (adir / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    (adir / "summary.md").write_text(render(out) + "\n")
    print(render(out))


def render(o):
    f = lambda x: f"{x:.1f}"  # noqa: E731
    md = [f"# e9 Adaptive-RAG baseline ({o['tag']}, test, 3 x 500 questions)", "",
          "EM / F1 in %; Tok = LLM tokens per question (prompt + completion) of the routed strategy only; "
          "Cost (×) = Tok / Standard RAG on the same dataset, avg = mean of the per-dataset ratios.", "",
          "| Method | " + " | ".join(f"{ds} EM | F1 | Tok" for ds in DATASETS) + " | avg EM | avg F1 | Cost (×) avg |",
          "|---" * (1 + 3 * len(DATASETS) + 3) + "|"]
    for m in o["methods"].values():
        md.append(f"| {m['name']} | " + " | ".join(f"{f(m[ds]['em'])} | {f(m[ds]['f1'])} | {m[ds]['tok']:,.0f}"
                                                   for ds in DATASETS)
                  + f" | {f(m['avg']['em'])} | {f(m['avg']['f1'])} | {m['avg']['cost_x']:.2f} |")
    for p in o["pending"]:
        md.append(f"| {p.split(':')[0]} | pending: {p.split(': ', 1)[1]} |" + " |" * (3 * len(DATASETS) + 2))
    md += ["", "## Routing (share of test questions sent to each strategy)", ""]
    for key, m in o["methods"].items():
        if "route_frac" in m:
            md.append(f"- {m['name']}: " + "; ".join(
                f"{ds} " + ", ".join(f"{l} {100 * v:.0f}%" for l, v in m["route_frac"][ds].items()) for ds in DATASETS))
    md += ["", "## Label distributions (label/source counts; source silver = some strategy correct, inductive = none "
               "correct -> C, default = neither correct -> CHEAP)", ""]
    for lset, d in o["labels"].items():
        for part, per in d.items():
            md.append(f"- {lset} {part}: " + "; ".join(
                f"{ds} " + ", ".join(f"{k} {v}" for k, v in sorted(per[ds].items())) for ds in DATASETS))
    md += ["", "## Classifier accuracy against test silver labels (%)", "",
           "| Label set | " + " | ".join(f"{ds} all | silver only" for ds in DATASETS + ("pooled",)) + " |",
           "|---" * (1 + 2 * (len(DATASETS) + 1)) + "|"]
    for lset, acc in o["classifier"].items():
        md.append(f"| {lset} | " + " | ".join(f"{f(acc[ds]['acc_all'])} | {f(acc[ds]['acc_silver_only'])} "
                                              f"(n={acc[ds]['n_silver_only']})" for ds in DATASETS + ("all",)) + " |")
    for lset, acc in o["classifier"].items():
        md.append(f"- {lset} confusion (all datasets, gold -> predicted counts): " + "; ".join(
            f"{g}: " + ", ".join(f"{p} {n}" for p, n in row.items()) for g, row in acc["all"]["confusion"].items()))
    if o["train"]:
        md += ["", "## Training", ""]
        for lset, t in o["train"].items():
            md.append(f"- {lset}: {t['n_train']} questions, {t['epochs']} epochs ({t['steps']} steps, batch "
                      f"{t['batch_size']}, lr {t['lr']}), final loss {t['history'][-1]['loss']:.4f}, training accuracy "
                      f"{100 * t['train_acc']:.1f}%, {t['train_seconds_total'] / 60:.0f} min on {t.get('device', 'cpu')}" + (f" ({t['threads']} CPU threads)" if t.get('device', 'cpu') == 'cpu' else ""))
    md += ["", "## Deviations from the official Adaptive-RAG recipe", ""] + [f"- {d}" for d in o["deviations"]]
    return "\n".join(md)


if __name__ == "__main__":
    main()
