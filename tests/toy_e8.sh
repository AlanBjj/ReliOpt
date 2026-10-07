#!/usr/bin/env bash
# Local test of e8 (execution-design ablation) on the toy project of tests/toy_e1.sh: runs all seven variants and the
# summary, and checks that every variant writes every question, that untyped plans keep non-TXT steps (only execution
# changes), and that cmp_llm really changes how CMP steps run.
# Usage: bash tests/toy_e8.sh [tmp dir]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
T="${1:-$(mktemp -d)}"
bash "$ROOT/tests/toy_e1.sh" "$T" > "$T/toy_e1.out" 2>&1 || { cat "$T/toy_e1.out"; exit 1; }
cd "$T/proj"
for ds in 2wikimultihopqa musique; do
  cp "cache/ircot/processed_data/$ds/train.jsonl" "cache/ircot/processed_data/$ds/test_subsampled.jsonl"
  cp "results/e0/ids/${ds}_cal.txt" "results/e0/ids/${ds}_test.txt"
done
cp cache/ircot/raw_data/2wikimultihopqa/train.json cache/ircot/raw_data/2wikimultihopqa/dev.json
mkdir -p results/e3/llama31-8b
echo '{"txt": "span", "k": 8, "sink_q": true}' > results/e3/llama31-8b/config.json
"$PY" "$ROOT/tests/fake_vllm.py" 8199 e1 > "$T/fake_vllm_e8.log" 2>&1 &
FAKE=$!
trap 'kill $FAKE 2>/dev/null' EXIT
sleep 1
"$PY" -u scripts/e8_exec_ablation.py --tag llama31-8b --gpus 99 --datasets 2wikimultihopqa musique --workers 8
"$PY" -u scripts/e8_exec_ablation.py --tag llama31-8b --gpus 99 --datasets 2wikimultihopqa musique --workers 8 | grep -c skipped
"$PY" scripts/e8_summarize.py --tag llama31-8b > /dev/null
"$PY" - <<'EOF'
import json
from pathlib import Path
base = Path("results/e8/llama31-8b")
for ds in ("2wikimultihopqa", "musique"):
    rows = {v: [json.loads(l) for l in (base / ds / f"{v}.jsonl").open()]
            for v in ("full", "no_sink", "short", "k5", "cmp_llm", "untyped", "no_check")}
    assert all(len(r) == 6 for r in rows.values()), {v: len(r) for v, r in rows.items()}
    assert not any("error" in x for r in rows.values() for x in r), [x["error"] for r in rows.values() for x in r if "error" in x][:1]
    kinds = {k for x in rows["untyped"] for k in x["kinds"]}
    assert kinds - {"TXT"}, kinds
    tok = {v: sum(x["tok"] for x in r) for v, r in rows.items()}
    has_cmp = any("CMP" in x["kinds"] for x in rows["full"])
    assert not has_cmp or tok["cmp_llm"] != tok["full"], tok
    assert tok["untyped"] != tok["full"] and tok["short"] != tok["full"], tok   # k5 can tie on the 9-passage toy corpus
    print(ds, "tokens", tok, "CMP plans:", has_cmp)
import sys; sys.path.insert(0, ".")
from scripts.e8_exec_ablation import VARIANTS
from reliopt.exec.replay import Pipeline
cfg = {"k": 8, "txt": "span", "sink_q": True}
pipes = {v: Pipeline(None, None, **{**cfg, **o}) for v, o in VARIANTS.items()}
assert pipes["k5"].ex.k == 5 and pipes["full"].ex.k == 8 and not pipes["no_sink"].ex.sink_q and pipes["short"].ex.txt == "short"
assert pipes["cmp_llm"].ex.exec_kind("CMP") == "LOG" and pipes["untyped"].ex.exec_kind("LOG") == "TXT" and not pipes["no_check"].strict
assert pipes["full"].ex.exec_kind("CMP") == "CMP" and pipes["full"].strict
s = json.loads((base / "summary.json").read_text())
assert set(s["musique"]) >= {"full", "untyped"} and "d_em" in s["musique"]["untyped"], s["musique"].keys()
print("[toy_e8] OK")
EOF
