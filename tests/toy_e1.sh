#!/usr/bin/env bash
# Local end-to-end test of the e1 pipeline with a fake vLLM server and a toy corpus; no GPU and no real data needed.
# Copies the package, scripts/ and run/ to a temporary directory, builds a toy corpus, questions and BM25 indexes there,
# starts tests/fake_vllm.py (port 8199) and runs scripts/e1_pilot.py (--n 5) and scripts/e1_analyze.py. The fake model
# is almost never right, so EM is 0: the test only checks that the pipeline runs, the record format and resuming.
# Usage: bash tests/toy_e1.sh [tmp dir]      (needs bm25s, PyStemmer and requests)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
T="${1:-$(mktemp -d)}"
rm -rf "$T/proj" && mkdir -p "$T/proj"
cp -R "$ROOT/reliopt" "$ROOT/scripts" "$ROOT/run" "$T/proj/"
find "$T/proj" -name __pycache__ -prune -exec rm -rf {} +
cd "$T/proj"
"$PY" - <<'EOF'
import json, sys
from pathlib import Path
corpus = [
 ("Alpha Corp", "Alpha Corp was founded in 1921 by engineers in Ohio."),
 ("Beta Inc", "Beta Inc was founded in 1930 as a food company."),
 ("Alpha Labs", "Alpha Labs started in 1950 and is unrelated."),
 ("Gamma Film", "The film Gamma Film was directed by John Smith in 1980."),
 ("John Smith", "John Smith was born in Paris City and directed films."),
 ("Delta Film", "Delta Film is a drama directed by Mary Jones."),
 ("Mary Jones", "Mary Jones was born in Rome Town in 1950."),
 ("Epsilon Band", "Epsilon Band released records with Kappa Label in 1999."),
 ("Kappa Label", "Kappa Label was founded by Omar Khan in 1970."),
]
qs = [
 ("q1", "Which was founded earlier, Alpha Corp or Beta Inc?", "Alpha Corp", "comparison",
  [["Alpha Corp", "inception", "1921"], ["Beta Inc", "inception", "1930"]]),
 ("q2", "Where was the director of Gamma Film born?", "Paris City", "compositional",
  [["Gamma Film", "director", "John Smith"], ["John Smith", "place of birth", "Paris City"]]),
 ("q3", "Are Gamma Film and Delta Film directed by the same person?", "no", "comparison",
  [["Gamma Film", "director", "John Smith"], ["Delta Film", "director", "Mary Jones"]]),
 ("q4", "BADPLAN Who founded the label of Epsilon Band?", "Omar Khan", "compositional",
  [["Epsilon Band", "record label", "Kappa Label"], ["Kappa Label", "founded by", "Omar Khan"]]),
 ("q5", "Which was founded earlier, Beta Inc or Alpha Labs?", "Beta Inc", "comparison",
  [["Beta Inc", "inception", "1930"], ["Alpha Labs", "inception", "1950"]]),
 ("q6", "Where was the director of Delta Film born?", "Rome Town", "compositional",
  [["Delta Film", "director", "Mary Jones"], ["Mary Jones", "place of birth", "Rome Town"]]),
]
for ds in ("2wikimultihopqa", "musique"):
    d = Path("cache/ircot/corpus"); d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{ds}.jsonl", "w") as f:
        for i, (t, x) in enumerate(corpus):
            f.write(json.dumps({"id": f"{ds}-{i}", "title": t, "text": x}) + "\n")
    d = Path(f"cache/ircot/processed_data/{ds}"); d.mkdir(parents=True, exist_ok=True)
    ids = []
    with open(d / "train.jsonl", "w") as f:
        for qid, q, ans, typ, ev in qs:
            qid2 = qid if ds == "2wikimultihopqa" else f"2hop__{qid}"
            ids.append(qid2)
            steps = [f"{s} >> {r} >>>> {o}" for s, r, o in ev]
            f.write(json.dumps({"question_id": qid2, "question_text": q,
                                "answers_objects": [{"number": "", "date": {"day": "", "month": "", "year": ""}, "spans": [ans]}],
                                "contexts": [], "reasoning_steps": steps}) + "\n")
    d = Path("results/e0/ids"); d.mkdir(parents=True, exist_ok=True)
    (d / f"{ds}_cal.txt").write_text("\n".join(sorted(ids)) + "\n")
d = Path("cache/ircot/raw_data/2wikimultihopqa"); d.mkdir(parents=True, exist_ok=True)
json.dump([{"_id": qid, "type": typ, "evidences": ev, "question": q, "answer": ans, "context": []}
           for qid, q, ans, typ, ev in qs], open(d / "train.json", "w"))
sys.path.insert(0, ".")
from scripts.e0_build_bm25 import build
for ds in ("2wikimultihopqa", "musique"):
    build(ds)
EOF
"$PY" "$ROOT/tests/fake_vllm.py" 8199 e1 > "$T/fake_vllm.log" 2>&1 &
FAKE=$!
trap 'kill $FAKE 2>/dev/null' EXIT
sleep 1
"$PY" -u scripts/e1_pilot.py --tag llama31-8b --gpus 99 --n 5 --workers 8
"$PY" -u scripts/e1_analyze.py > /dev/null
echo "[toy_e1] OK: $T/proj/results/e1/summary.md"
