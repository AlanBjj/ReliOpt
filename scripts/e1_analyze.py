"""e1 pilot analysis: headroom, oracle frontier, type-level policies, and where the value of verification sits.

Input:  results/e1/<ds>/questions.jsonl (scripts/e1_pilot.py)
Output: results/e1/summary.json, results/e1/summary.md
Usage:  python scripts/e1_analyze.py [--datasets 2wikimultihopqa musique] [--run r2]   (outputs go next to the rows)
CPU only, seconds; overwrites its two outputs. Costs are verification tokens per question (prompt + completion of every
verification call a policy would issue); EM is in percent.

Go/no-go criteria:
  C1 Verify-All beats Plan-Only by >= 3 EM on one dataset and >= 1.5 on the other.
  C2 The oracle reaches >= 80% of that gap with <= half of Verify-All's verification tokens.
  C3 At that cost the oracle beats type-level static allocation by >= 1 EM.
Verify-All is reported under two definitions (the composite VOTE5+ALT on every step; the best assignment of one single
operator, VOTE3 / VOTE5 / ALT, to each type, chosen in-sample),
Type-Static under two (the fixed point TXT ALT / CMP ALT / LOG VOTE3; the best of all 125 type-level policies at
the same cost, chosen in-sample and therefore favourable to the baseline). The oracle picks, per question, the action with
the largest realised EM minus lambda x cost among: no verification, one operator at one step ("single"), and additionally
every type-level policy ("all").
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reliopt.quality.features import FEATURES  # noqa: E402

VERIFY_ALL = "VOTE5+ALT|VOTE5+ALT|VOTE5+ALT"
TYPE_STATIC = "ALT|ALT|VOTE3"
VERIFY_ALL_DEEP = "DEEP|DEEP|DEEP"   # DEEPGROUND on TXT, VOTE5+REROUTE on CMP / LOG (rounds with DEEPGROUND only)
NAMED = {"plan_only": "NONE|NONE|NONE", "verify_all": VERIFY_ALL, "type_static": TYPE_STATIC,
         "all_vote3": "VOTE3|VOTE3|VOTE3", "all_vote5": "VOTE5|VOTE5|VOTE5", "all_alt": "ALT|ALT|ALT",
         "verify_all_deep": VERIFY_ALL_DEEP}
SINGLE_OPS = ("VOTE3", "VOTE5", "ALT")   # "best single": every type gets one single operator


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def r1(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, 1)


def r3(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, 3)


def policy_point(rows, name):
    return {"em": 100 * mean(r["grid"][name]["em"] for r in rows), "f1": 100 * mean(r["grid"][name]["f1"] for r in rows),
            "verify_tok": mean(r["grid"][name]["tok"]["verify"] for r in rows),
            "total_tok": mean(sum(r["grid"][name]["tok"].values()) for r in rows)}


def actions(r, use_grid):
    """(em, f1, verify tokens) of every cached action for one question."""
    acts = [(r["base"]["em"], r["base"]["f1"], 0)]
    acts += [(v["em"], v["f1"], v["tok"]["verify"]) for v in r["single"].values()]
    if use_grid:
        acts += [(v["em"], v["f1"], v["tok"]["verify"]) for v in r["grid"].values()]
    return acts


def oracle_frontier(rows, use_grid):
    acts = [actions(r, use_grid) for r in rows]
    lams = [0.0] + [10 ** (-8 + 8 * i / 400) for i in range(401)]
    pts = {}
    for lam in lams:
        em = f1 = tok = 0.0
        for a in acts:
            best = max(a, key=lambda x: (x[0] - lam * x[2], -x[2], x[1]))
            em, f1, tok = em + best[0], f1 + best[1], tok + best[2]
        n = len(acts)
        pts[(round(tok / n, 3), round(100 * em / n, 3))] = 100 * f1 / n
    out = sorted(({"verify_tok": k[0], "em": k[1], "f1": v} for k, v in pts.items()),
                 key=lambda p: (p["verify_tok"], p["em"]))
    front, best = [], -1   # keep the Pareto frontier
    for p in out:
        if p["em"] > best:
            front.append(p); best = p["em"]
    return front


def em_at(points, cost):
    """Best EM among points whose cost does not exceed `cost` (step function, never interpolated)."""
    ok = [p["em"] for p in points if p["verify_tok"] <= cost + 1e-9]
    return max(ok) if ok else float("nan")


def cost_for(points, em_target):
    ok = [p["verify_tok"] for p in points if p["em"] >= em_target - 1e-9]
    return min(ok) if ok else float("inf")


def auroc(scores, ys):
    pos = [s for s, y in zip(scores, ys) if y]
    neg = [s for s, y in zip(scores, ys) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def cv_auroc(X, y, folds=5):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None
    if sum(y) < folds or len(y) - sum(y) < folds:
        return None
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    p = cross_val_predict(model, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=0), method="predict_proba")
    return auroc(list(p[:, 1]), y)


def analyse(ds, rows):
    n = len(rows)
    res = {"n": n}
    # ---- plans
    kinds = Counter(s["kind"] for r in rows for s in r["plan"]["steps"])
    res["plans"] = {"fallback_rate": 100 * mean(r["plan"]["fallback"] for r in rows),
                    "retry_rate": 100 * mean(r["plan"]["attempts"] > 1 for r in rows),
                    "mean_steps": mean(len(r["plan"]["steps"]) for r in rows),
                    "steps_hist": dict(sorted(Counter(len(r["plan"]["steps"]) for r in rows).items())),
                    "kinds": dict(kinds), "plan_tok": mean(r["plan"]["tok"] for r in rows),
                    "cmp_exec_fail_rate": 100 * mean(not s.get("exec_ok") for r in rows for s in r["steps"]
                                                     if s["kind"] == "CMP") if kinds.get("CMP") else None}
    # ---- policies
    res["plan_only"] = {"em": 100 * mean(r["base"]["em"] for r in rows), "f1": 100 * mean(r["base"]["f1"] for r in rows),
                        "exec_tok": mean(r["base"]["tok"]["exec"] for r in rows),
                        "total_tok": mean(sum(r["base"]["tok"].values()) for r in rows)}
    res["policies"] = {k: policy_point(rows, v) for k, v in NAMED.items() if v in rows[0]["grid"]}
    grid = {name: policy_point(rows, name) for name in rows[0]["grid"]}
    singles = {k: v for k, v in grid.items() if all(o in SINGLE_OPS for o in k.split("|"))}
    best_single = max(singles, key=lambda k: (singles[k]["em"], -singles[k]["verify_tok"]))
    res["policies"]["verify_all_best_single"] = {"name": best_single, **singles[best_single]}
    res["grid"] = {k: {"em": r1(v["em"]), "verify_tok": r1(v["verify_tok"])} for k, v in grid.items()}
    grid_pts = [{"verify_tok": v["verify_tok"], "em": v["em"]} for v in grid.values()]
    fr_single, fr_all = oracle_frontier(rows, False), oracle_frontier(rows, True)
    res["oracle_single"] = [{k: r1(v) for k, v in p.items()} for p in fr_single]
    res["oracle_all"] = [{k: r1(v) for k, v in p.items()} for p in fr_all]
    # ---- criteria
    po = res["plan_only"]["em"]
    crit = {}
    vas = [("composite", res["policies"]["verify_all"]), ("best_single", res["policies"]["verify_all_best_single"])]
    if "verify_all_deep" in res["policies"]:
        vas.append(("deep", res["policies"]["verify_all_deep"]))
    for va_name, va in vas:
        gap = va["em"] - po
        half = 0.5 * va["verify_tok"]
        c = {"gap_em": gap, "verify_all_em": va["em"], "verify_all_tok": va["verify_tok"], "half_tok": half}
        for o_name, fr in (("single", fr_single), ("all", fr_all)):
            need = cost_for(fr, po + 0.8 * gap) if gap > 0 else float("nan")
            c2 = gap > 0 and need <= half
            at = need if c2 else half
            ts = res["policies"]["type_static"]
            c[f"oracle_{o_name}"] = {
                "tok_for_80pct_gap": need, "C2_pass": bool(c2), "eval_cost": at,
                "oracle_em_at_cost": em_at(fr, at), "typestatic_grid_em_at_cost": em_at(grid_pts, at),
                "C3_margin_vs_grid": em_at(fr, at) - em_at(grid_pts, at),
                "oracle_em_at_typestatic_cost": em_at(fr, ts["verify_tok"]),
                "C3_margin_vs_fixed": em_at(fr, ts["verify_tok"]) - ts["em"]}
            c[f"oracle_{o_name}"]["C3_pass_grid"] = c[f"oracle_{o_name}"]["C3_margin_vs_grid"] >= 1.0
            c[f"oracle_{o_name}"]["C3_pass_fixed"] = c[f"oracle_{o_name}"]["C3_margin_vs_fixed"] >= 1.0
        crit[va_name] = c
    res["criteria"] = crit
    # ---- step labels, operator repair / damage, realised single-step gains
    steps = [(r, s) for r in rows for s in r["steps"]]
    lab = defaultdict(Counter)
    for _, s in steps:
        lab[s["kind"]][s["label"]] += 1
    res["labels"] = {k: dict(v) for k, v in lab.items()}
    res["labels"]["undet_rate"] = 100 * mean(s["label"] == "undet" for _, s in steps)
    det = [(r, s) for r, s in steps if s["label"] != "undet"]
    by = defaultdict(list)
    for r, s in det:
        pos = "sink" if s["id"] == r["plan"]["steps"][-1]["id"] else f"depth{int(s['feat']['depth'])}"
        by[f"{s['kind']}/{pos}"].append(s["label"] == "error")
    res["step_error_rate"] = {k: {"n": len(v), "err": r1(100 * mean(v))} for k, v in sorted(by.items())}
    rho = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))   # fixed, n_err, damaged, n_ok
    for _, s in det:
        for op, o in s["ops"].items():
            if op == "PROBE" or o["label"] == "undet":
                continue
            c = rho[s["kind"]][op]
            if s["label"] == "error":
                c[0] += o["label"] == "correct"; c[1] += 1
            else:
                c[2] += o["label"] == "error"; c[3] += 1
    res["rho"] = {k: {op: {"rho_plus": r1(100 * c[0] / c[1]) if c[1] else None, "n_err": c[1],
                           "rho_minus": r1(100 * c[2] / c[3]) if c[3] else None, "n_ok": c[3]}
                      for op, c in v.items()} for k, v in rho.items()}
    gains = defaultdict(lambda: Counter())
    for r, s in steps:
        g = [v["em"] - r["base"]["em"] for k, v in r["single"].items() if k.startswith(f"{s['id']}:")]
        pos = "sink" if s["id"] == r["plan"]["steps"][-1]["id"] else "inner"
        key = f"{s['kind']}/{pos}"
        gains[key]["steps"] += 1
        gains[key]["fixable"] += any(x > 0 for x in g)
        gains[key]["breakable"] += any(x < 0 for x in g)
    res["single_step_gains"] = {k: {"steps": v["steps"], "fixable_pct": r1(100 * v["fixable"] / v["steps"]),
                                    "breakable_pct": r1(100 * v["breakable"] / v["steps"])} for k, v in gains.items()}
    wrong = [r for r in rows if r["base"]["em"] == 0]
    fixable_q = [r for r in wrong if any(v["em"] > 0 for v in r["single"].values())]
    n_fix_steps = [len({k.split(":")[0] for k, v in r["single"].items() if v["em"] > 0}) for r in fixable_q]
    res["questions"] = {"wrong": len(wrong), "fixable_by_one_step": len(fixable_q),
                        "fixing_steps_per_fixable_q": r3(mean(n_fix_steps)),
                        "steps_per_fixable_q": r3(mean(len(r["steps"]) for r in fixable_q))}
    # ---- probe agreement and error predictability
    agree = [(s["ops"]["PROBE"]["agree"], s["label"] == "error") for _, s in det]
    dis = [e for a, e in agree if not a]
    agr = [e for a, e in agree if a]
    res["probe"] = {"disagree_rate": r1(100 * len(dis) / len(agree)) if agree else None,
                    "err_if_disagree": r1(100 * mean(dis)), "err_if_agree": r1(100 * mean(agr))}
    ys = [int(s["label"] == "error") for _, s in det]
    single_feats = {"-lp_min": [-s["feat"]["lp_min"] for _, s in det], "-lp_mean": [-s["feat"]["lp_mean"] for _, s in det],
                    "not_found": [s["feat"]["not_found"] for _, s in det],
                    "-in_cited": [-s["feat"]["in_cited"] for _, s in det],
                    "probe_disagree": [float(not s["ops"]["PROBE"]["agree"]) for _, s in det]}
    res["auroc"] = {k: r3(auroc(v, ys)) for k, v in single_feats.items()}
    X = [[s["feat"][f] for f in FEATURES] for _, s in det]
    res["auroc"]["logreg_free_cv5"] = r3(cv_auroc(X, ys))
    X2 = [x + [float(not s["ops"]["PROBE"]["agree"])] for x, (_, s) in zip(X, det)]
    res["auroc"]["logreg_free+probe_cv5"] = r3(cv_auroc(X2, ys))
    res["auroc"]["n_steps"], res["auroc"]["n_err"] = len(ys), sum(ys)
    return res


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def markdown(summary):
    L = ["# e1 先导检验结果", "", "由 `scripts/e1_analyze.py` 生成。EM 为百分数；代价为每题平均验证 token（不含计划与默认执行）。", ""]
    dss = list(summary["datasets"])
    L += ["## 主要数字", "", "| 指标 | " + " | ".join(dss) + " |", "|---|" + "---|" * len(dss)]
    def row(name, f):
        L.append(f"| {name} | " + " | ".join(f(summary["datasets"][d]) for d in dss) + " |")
    row("题数", lambda r: str(r["n"]))
    row("计划退回单步 %", lambda r: fmt(r["plans"]["fallback_rate"]))
    row("平均步数", lambda r: fmt(r["plans"]["mean_steps"], 2))
    row("Plan-Only EM / F1", lambda r: f"{fmt(r['plan_only']['em'])} / {fmt(r['plan_only']['f1'])}")
    row("Plan-Only 总 token（计划+执行）", lambda r: fmt(r["plan_only"]["total_tok"], 0))
    for k in ("verify_all", "verify_all_deep", "verify_all_best_single", "type_static", "all_vote3", "all_vote5", "all_alt"):
        if all(k in summary["datasets"][d]["policies"] for d in dss):
            row(f"{k} EM（验证 token）", lambda r, k=k: f"{fmt(r['policies'][k]['em'])}（{fmt(r['policies'][k]['verify_tok'], 0)}）")
    row("oracle-single 上限 EM（验证 token）", lambda r: f"{fmt(r['oracle_single'][-1]['em'])}（{fmt(r['oracle_single'][-1]['verify_tok'], 0)}）")
    row("best single 的类型配置", lambda r: r["policies"]["verify_all_best_single"]["name"])
    L += ["", "## 通过标准", ""]
    for va in [v for v in ("composite", "best_single", "deep") if all(v in summary["datasets"][d]["criteria"] for d in dss)]:
        L += [f"### Verify-All = {va}", "", "| 项 | " + " | ".join(dss) + " |", "|---|" + "---|" * len(dss)]
        row("C1 差距 Verify-All − Plan-Only（EM）", lambda r, va=va: fmt(r["criteria"][va]["gap_em"]))
        for o in ("single", "all"):
            k = f"oracle_{o}"
            row(f"C2 oracle-{o} 达到 80% 差距所需 token / 全验证一半",
                lambda r, va=va, k=k: f"{fmt(r['criteria'][va][k]['tok_for_80pct_gap'], 0)} / "
                                      f"{fmt(r['criteria'][va]['half_tok'], 0)} → {'过' if r['criteria'][va][k]['C2_pass'] else '不过'}")
            row(f"C3 oracle-{o} − 类型静态最优（同代价，EM）",
                lambda r, va=va, k=k: f"{fmt(r['criteria'][va][k]['C3_margin_vs_grid'])} → "
                                      f"{'过' if r['criteria'][va][k]['C3_pass_grid'] else '不过'}")
            row(f"C3 oracle-{o} − Type-Static 固定点（在其代价处，EM）",
                lambda r, va=va, k=k: f"{fmt(r['criteria'][va][k]['C3_margin_vs_fixed'])} → "
                                      f"{'过' if r['criteria'][va][k]['C3_pass_fixed'] else '不过'}")
        L.append("")
    gaps = [summary["datasets"][d]["criteria"]["composite"]["gap_em"] for d in dss]
    if len(gaps) == 2:
        L.append(f"C1（composite）：最大差距 {fmt(max(gaps))}（需 ≥ 3），最小差距 {fmt(min(gaps))}（需 ≥ 1.5）→ "
                 f"{'过' if max(gaps) >= 3 and min(gaps) >= 1.5 else '不过'}")
    L += ["", "## 步骤层面", ""]
    for d in dss:
        r = summary["datasets"][d]
        L += [f"### {d}", "", f"- 标签：{json.dumps(r['labels'], ensure_ascii=False)}",
              f"- 步骤错误率（可判定步骤）：{json.dumps(r['step_error_rate'], ensure_ascii=False)}",
              f"- 单步干预能修好 / 会弄坏的步骤比例：{json.dumps(r['single_step_gains'], ensure_ascii=False)}",
              f"- 题目：{json.dumps(r['questions'], ensure_ascii=False)}",
              f"- PROBE：{json.dumps(r['probe'], ensure_ascii=False)}",
              f"- 出错可预测性 AUROC：{json.dumps(r['auroc'], ensure_ascii=False)}",
              f"- 修复率 / 误伤率 ρ±（%）：{json.dumps(r['rho'], ensure_ascii=False)}", ""]
    return "\n".join(L) + "\n"


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else round(o, 3)
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=["2wikimultihopqa", "musique"])
    ap.add_argument("--run", default="", help="round name used by e1_pilot.py --run")
    args = ap.parse_args()
    summary = {"datasets": {}}
    for ds in args.datasets:
        path = ROOT / "results" / "e1" / args.run / ds / "questions.jsonl"
        rows = [json.loads(l) for l in path.open()]
        summary["datasets"][ds] = analyse(ds, rows)
    summary = clean(summary)
    out = ROOT / "results" / "e1" / args.run
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (out / "summary.md").write_text(markdown(summary))
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
