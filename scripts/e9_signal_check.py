"""e9: which ablated escalation signals stay in the ablation table (rule fixed on 2026-10-08 01:10, before the
1000-question maps existed): a row (w/o propagation, type prior) is shown when the method's composed reliability does not
lose to it on average at equal spend; otherwise it is left out and the text claims nothing about which signal is better.
Equal spend: for each backbone and dataset, the mean over spends x = own + 1k, ..., own + 16k (own = the method's tokens at
B_SC) of the best EM a signal reaches with tokens <= x (its points over the budget sweep, E9_MODE); averaged over the six
backbone x dataset cells.
Input:  results/e9/<tag>/<E9_SRC>.json (make_table1.py)
Output: results/e9/signal_check.json and a printed table; the decision is copied into make_table_esc.py (ABLATED)
Usage:  python3 scripts/e9_signal_check.py      standard library only
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_table1 import E9_MODE, E9_SRC  # noqa: E402

TAGS = ["llama31-8b", "qwen3-8b"]
SIGS = ["plan", "noprop", "prior", "sink", "random"]


def area(r, sig, mode):
    own = r["own_tok"]
    pts = [(p["tok"], p["em"]) for k, p in r["points"].items() if k.startswith(f"{sig}/{mode}/")]
    grid = [own + 1000 * k for k in range(1, 17)]
    return sum(max([e for t, e in pts if t <= x + 1] or [r["refs"]["ours"]]) for x in grid) / len(grid)


def main():
    cells, out = {}, {"mode": E9_MODE, "src": E9_SRC, "cells": {}}
    for tag in TAGS:
        d = json.loads((ROOT / "results" / "e9" / tag / f"{E9_SRC}.json").read_text())
        for ds, r in d.items():
            cells[tag, ds] = {s: area(r, s, E9_MODE) for s in SIGS}
            out["cells"][f"{tag}/{ds}"] = cells[tag, ds]
    mean = {s: sum(c[s] for c in cells.values()) / len(cells) for s in SIGS}
    out["mean"] = mean
    out["keep"] = [s for s in ("noprop", "prior") if mean["plan"] >= mean[s]]
    print(f"{'cell':28s} " + " ".join(f"{s:>7s}" for s in SIGS))
    for (tag, ds), c in cells.items():
        print(f"{tag + '/' + ds:28s} " + " ".join(f"{c[s]:7.2f}" for s in SIGS))
    print(f"{'mean':28s} " + " ".join(f"{mean[s]:7.2f}" for s in SIGS))
    print("rows kept:", out["keep"])
    (ROOT / "results" / "e9" / "signal_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
