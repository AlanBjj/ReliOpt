"""Quality-cost curves: EM against tokens per question on each test set, one panel per dataset.

Series: the method (price sweep, total tokens incl. plan and execution, mean over seeds) and IRCoT (reduced budgets and the
official setting) as lines; the method with plan-level escalation (e9: the escalation price swept over budgets, plan signal,
E9_MODE of make_table1.py; from results/e9/<tag>/<ESC_SRC>_curve.csv) as a third line starting at the method's B_SC point;
Self-Consistency over k as a thin neutral line; the other baselines as labelled points.
Colours: blue (the method), orange (IRCoT) and aqua (with escalation) are the first three categorical slots of the dataviz
reference palette (validated 2026-10-08: worst adjacent CVD dE 9.2, normal 27.6; aqua is 2.74:1 against the surface, so its
line also carries its own marker, a legend entry and a direct label); everything else is neutral and labelled directly.
Input:  results/e2/<tag>/test/summary.json, results/e3/<tag>/test/summary.json, results/e9/<tag>/<ESC_SRC>_curve.csv
Output: results/figures/qc_<tag>.png (preview) and results/figures/qc_<tag>.csv (the plotted numbers, for pgfplots)
Usage:  python scripts/make_fig_qc.py --tag llama31-8b       needs matplotlib (local venv)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_table1 import E9_MODE, E9_SRC as ESC_SRC  # noqa: E402   one escalation setting for the main table and the figure
DATASETS = [("hotpotqa", "HotpotQA"), ("2wikimultihopqa", "2WikiMultiHopQA"), ("musique", "MuSiQue")]
OURS, IRCOT, ESC, NEUTRAL, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8f98", "#1f2328", "#5b616b"

LABEL_OFFSET = {"pyrag": (5, -10)}   # keeps PyRAG's label clear of Plan-and-Budget's where the two points meet
POINTS = [("rag", "Standard RAG"), ("ac", "Adaptive-Cons."), ("selfask", "Self-Ask"), ("searcho1", "Search-o1"),
          ("pyrag", "PyRAG"), ("planbudget", "Plan-and-Budget"), ("arag", "Adaptive-RAG")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="llama31-8b")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e2 = json.loads((ROOT / "results" / "e2" / args.tag / "test" / "summary.json").read_text())
    e3 = json.loads((ROOT / "results" / "e3" / args.tag / "test" / "summary.json").read_text())
    # Adaptive-RAG with its official recipe (the main table has it at a matched budget)
    ap_ = ROOT / "results" / "e9" / args.tag / "arag" / "summary.json"
    arag = json.loads(ap_.read_text())["methods"]["arag"] if ap_.exists() else {}
    for ds in arag:
        if ds in e2["summary"]:
            e2["summary"][ds]["arag"] = arag[ds]
    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    esc_all = {}
    cp = ROOT / "results" / "e9" / args.tag / f"{ESC_SRC}_curve.csv"
    if cp.exists():
        for r in csv.DictReader(cp.open()):
            if r["signal"] == "plan" and r["mode"] == E9_MODE:
                esc_all.setdefault(r["dataset"], set()).add((round(float(r["tok"]), 1), round(float(r["em"]), 2)))
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (ds, name) in zip(axes, DATASETS):
        s2 = e2["summary"].get(ds, {})
        ours = sorted((p["total_tok"], p["em"]) for p in e3.get(ds, {}).get("curves", {}).get("full", []))
        irc = sorted([(v["tok"], v["em"]) for k, v in s2.items() if k.startswith("ircotb_")]
                     + ([(s2["ircot"]["tok"], s2["ircot"]["em"])] if "ircot" in s2 else []))
        sc = [(p["tok"], p["em"]) for k, p in sorted(e2["sc_curve"].get(ds, {}).items(), key=lambda kv: int(kv[0]))]
        if sc:
            ax.plot(*zip(*sc), color=NEUTRAL, lw=1.2, alpha=0.7, zorder=1)
            ax.annotate("Self-Consistency (k=1..10)", sc[0], textcoords="offset points", xytext=(0, -11), fontsize=7,
                        color=MUTED)
        if irc:
            ax.plot(*zip(*irc), color=IRCOT, lw=2, marker="o", ms=5, mec="white", mew=1.2, label="IRCoT (budget sweep)",
                    zorder=3)
        if ours:
            ax.plot(*zip(*ours), color=OURS, lw=2, marker="o", ms=3, mec=OURS, mew=0, label="Ours (price sweep)",
                    zorder=4)
        esc = sorted(esc_all.get(ds, ()))
        if esc:
            ax.plot(*zip(*esc), color=ESC, lw=2, marker="^", ms=5, mec="white", mew=1, label="Ours + escalation (price sweep)",
                    zorder=5)
            ax.annotate("+ esc.", esc[-1], textcoords="offset points", xytext=(4, 2), fontsize=7, color=MUTED)
        for key, label in POINTS:
            if key in s2:
                x, y = s2[key]["tok"], s2[key]["em"]
                ax.scatter([x], [y], s=28, color=NEUTRAL, edgecolor="white", linewidth=1, zorder=2)
                ax.annotate(label, (x, y), textcoords="offset points", xytext=LABEL_OFFSET.get(key, (5, 3)), fontsize=7,
                            color=MUTED)
                rows.append([ds, label, round(x), round(y, 2)])
        for series, pts in (("ours", ours), ("ircot", irc), ("sc", sc), ("esc", esc)):
            rows += [[ds, series, round(x), round(y, 2)] for x, y in pts]
        ax.set_xscale("log")
        ax.set_title(name, loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.set_xlabel("tokens per question (log scale)", fontsize=8, color=MUTED)
        ax.grid(True, which="major", color="#e6e8eb", lw=0.6)
        ax.tick_params(labelsize=7, colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c9cdd2")
    axes[0].set_ylabel("EM (%)", fontsize=8, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / f"qc_{args.tag}.png", dpi=180, facecolor="#fcfcfb")
    with open(out / f"qc_{args.tag}.csv", "w", newline="") as f:
        csv.writer(f).writerows([["dataset", "series", "tokens", "em"]] + rows)
    print(f"wrote {out / f'qc_{args.tag}.png'}")


if __name__ == "__main__":
    main()
