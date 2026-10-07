"""Table 1 (main results) of the paper, generated from the e2 and e3 summaries; never edit the .tex by hand.

Layout: one block per backbone; per dataset EM and F1; one row for the
method (escalation at B_IRCoT/2); IRCoT and Adaptive-RAG at a budget matched to that row (closest tokens
per question; their official settings are in the quality-cost figure); the last column is
Cost (x) = tokens per question relative to Standard RAG, averaged over the three datasets, printed with a bar (\\cbar, no tint).
EM / F1 cells are tinted by their change against Standard RAG (\\hc of macros.tex: blue better, orange worse; the key is
\\tintlegend in the section file); best in bold and second underlined per block
and column (EM / F1 only). Sampled methods show the mean over seeds. Missing numbers print as \\tbd.
Input:  results/e2/<tag>/test/summary.json, results/e3/<tag>/test/summary.json ("main": the method at the priced lambdas),
        results/e9/<tag>/arag/summary.json (Adaptive-RAG),
        results/e9/<tag>/<E9_SRC>.json (the method with plan-level escalation; rows are added only when it exists)
Output: paper_arxiv/tables/table1_main.tex (a tabular; caption and float live in the section file)
Usage:  python scripts/make_table1.py [--tags llama31-8b qwen3-8b]      standard library only
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = [("hotpotqa", "HotpotQA"), ("2wikimultihopqa", "2WikiMultiHopQA"), ("musique", "MuSiQue")]
BACKBONES = {"llama31-8b": "Llama-3.1-8B-Instruct", "qwen3-8b": "Qwen3-8B"}
ROWS = [("cot", "CoT"), ("rag", "Standard RAG"), ("sc5", "Self-Consistency"), ("ac", "Adaptive-Consistency"),
        ("selfask", "Self-Ask"), ("ircot_matched", r"IRCoT (matched budget)"), ("arag", r"Adaptive-RAG (matched budget)"),
        ("searcho1", "Search-o1"),
        ("pyrag", "PyRAG"), ("planbudget", "Plan-and-Budget")]
# IRCoT in the main table runs at a budget matched to the method: per dataset, the reduced-budget
# configuration (scripts/e2_baselines.py ircot_budget) whose tokens per question are closest to the method at B_SC;
# the official IRCoT goes to the appendix table and the quality-cost figure.
# One row for the method: the full method, escalation at the pre-set total budget B_IRCoT/2; the
# verification-only method and the other budgets are in the escalation table and the quality-cost figure.
OURS = []
# With plan-level escalation (e9): the method at total budgets anchored to measured baseline costs,
# Search-o1's (B_SO1) and half of official IRCoT's (B_IRCoT/2), read from results/e9/<tag>/priced.json. E9_MODE says how
# the escalation price was set (scripts/e9_price.py: "cal" or "workload").
E9_MODE = "workload"
# Which escalation maps: priced_dev1000 = 1000 development questions (devcal + devcal2), Search-o1 labels averaged over
# seeds 17 / 29 / 43
E9_SRC = "priced_dev1000"
OURS_E9 = [("B_IRCoT/2", r"\method{}")]
MAIN_BUDGET = "B_IRCoT/2"   # the budget the matched-budget baselines (IRCoT, Adaptive-RAG) are matched to


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def block(tag):
    e2 = load(ROOT / "results" / "e2" / tag / "test" / "summary.json").get("summary", {})
    e3 = load(ROOT / "results" / "e3" / tag / "test" / "summary.json")
    e9 = load(ROOT / "results" / "e9" / tag / f"{E9_SRC}.json")
    for ds, _ in DATASETS:   # the matched-budget IRCoT entry: closest tokens to the method's main-table row
        ours = e9.get(ds, {}).get("points", {}).get(f"plan/{E9_MODE}/{MAIN_BUDGET}")
        cands = {k: v for k, v in e2.get(ds, {}).items() if k.startswith("ircotb_")}
        if ours and cands:
            k = min(cands, key=lambda c: abs(cands[c]["tok"] - ours["tok"]))
            e2[ds]["ircot_matched"] = {**cands[k], "cfg": k}
    # Adaptive-RAG (e9, a required baseline:) at a matched budget like IRCoT:
    # results/e9/<tag>/arag/matched.json (scripts/e9_arag_matched.py); cost_x is relative to Standard RAG on the dataset
    arag = load(ROOT / "results" / "e9" / tag / "arag" / "matched.json")
    for ds, _ in DATASETS:
        if ds in arag:
            e2.setdefault(ds, {})["arag"] = {k: arag[ds][k] for k in ("em", "f1", "tok", "cost_x")}
    table = []   # (label, is_ours, {ds: (em, f1)}, cost_x or None)
    for key, label in ROWS:
        vals = {ds: (e2[ds][key]["em"], e2[ds][key]["f1"]) for ds, _ in DATASETS if key in e2.get(ds, {})}
        costs = [e2[ds][key]["cost_x"] for ds, _ in DATASETS if key in e2.get(ds, {}) and "cost_x" in e2[ds][key]]
        table.append((label, False, vals, sum(costs) / len(costs) if len(costs) == len(DATASETS) else None))
    for b, label in OURS:
        vals = {ds: (e3[ds]["main"][b]["em"], e3[ds]["main"][b]["f1"]) for ds, _ in DATASETS
                if b in e3.get(ds, {}).get("main", {})}
        costs = [e3[ds]["main"][b]["cost_x"] for ds, _ in DATASETS if b in e3.get(ds, {}).get("main", {})]
        table.append((label, True, vals, sum(costs) / len(costs) if len(costs) == len(DATASETS) else None))
    for b, label in OURS_E9:
        pts = {ds: e9[ds]["points"].get(f"plan/{E9_MODE}/{b}") for ds, _ in DATASETS if ds in e9}
        vals = {ds: (p["em"], p["f1"]) for ds, p in pts.items() if p}
        costs = [pts[ds]["tok"] / e2[ds]["rag"]["tok"] for ds, _ in DATASETS if pts.get(ds) and "rag" in e2.get(ds, {})]
        if vals:
            table.append((label, True, vals, sum(costs) / len(costs) if len(costs) == len(DATASETS) else None))
    ref = {ds: e2[ds]["rag"] for ds, _ in DATASETS if "rag" in e2.get(ds, {})}
    return table, ref


def ranks(table, ds, i):
    vals = sorted({round(v[ds][i], 1) for _, _, v, _ in table if ds in v}, reverse=True)
    return vals[:1], vals[1:2]


def render(tags):
    out = [r"% Generated by scripts/make_table1.py from results/e2/<tag>/test/summary.json and results/e3/<tag>/test/summary.json.",
           r"% Do not edit by hand.",
           r"\begin{tabular}{@{}l" + "cc" * len(DATASETS) + "c@{}}", r"\toprule",
           " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{name}}}" for _, name in DATASETS) + r" & \\",
           "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DATASETS))),
           "Method & " + " & ".join(r"EM $\uparrow$ & F1 $\uparrow$" for _ in DATASETS) + r" & Cost ($\times$) $\downarrow$ \\",
           r"\midrule"]
    blocks = [block(tag) for tag in tags]
    cmax = max([c for table, _ in blocks for _, _, _, c in table if c is not None] or [1.0])   # bar scale: largest cost
    for t, tag in enumerate(tags):
        table, ref = blocks[t]
        if t:
            out.append(r"\midrule")
        out.append(rf"\grp{{{1 + 2 * len(DATASETS) + 1}}}{{{BACKBONES.get(tag, tag)}}} \\")
        for label, ours, vals, cost in table:
            cells = []
            for ds, _ in DATASETS:
                for i, metric in ((0, "em"), (1, "f1")):
                    if ds not in vals:
                        cells.append(r"\tbd")
                        continue
                    v = round(vals[ds][i], 1)
                    best, second = ranks(table, ds, i)
                    fmt = r"[\textbf]" if v in best else r"[\underline]" if v in second else ""
                    cells.append(rf"\hc{fmt}{{{v:.1f}}}{{{ref[ds][metric]:.1f}}}" if ds in ref else f"{v:.1f}")
            cells.append(r"\tbd" if cost is None else rf"\cbar{{{cost:.2f}}}{{{cmax:.2f}}}")   # cost: a bar, no tint
            out.append((r"\ours " if ours else r"\base ") + label + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags", nargs="+", default=list(BACKBONES))
    args = ap.parse_args()
    path = ROOT / "results" / "tables" / "table1_main.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(args.tags))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
