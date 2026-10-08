# ReliOpt

Code for the paper **"ReliOpt: Progressive, Cost-Based Optimization for Reliable Compositional Question Answering"** (Jie Bao, Haojie Wu, Xiaolu Chen, Yu Gao, Zhen Chen, Yong Liao; submitted to EDBT).

ReliOpt treats where to spend tokens for reliability in compositional question answering as cost-based optimization
over a per-question query plan. The cheap plan runs first; what its steps leave behind decides whether to accept the
answer, verify a weak step, or re-run the whole question with an expensive strategy.

* **Typed plan.** A frozen LLM decomposes the question into a small dependency graph of TXT (retrieve and read), CMP
  (a short program) and LOG steps, executed in order over BM25 retrieval.
* **Quality model.** Light estimators fitted on calibration questions predict how likely each step is wrong, what each
  verification operator would repair or damage, and how an error reaches the answer, and compose them into the
  reliability of the answer.
* **Two prices.** After the default execution, the reliability is mapped to calibrated probabilities that the plan and
  each alternative strategy (IRCoT, Search-o1) answer correctly. A question is re-executed with an alternative where the
  expected gain pays for its tokens at the escalation price, set by the remaining budget; the other questions receive
  step-level verification at a price calibrated beforehand.

## Main results (EM on the IRCoT test subsets, same backbone and retriever for every method)

| Backbone | Method | HotpotQA | 2Wiki | MuSiQue | Cost (×) |
|---|---|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | Standard RAG | 36.0 | 27.6 | 10.0 | 1.00 |
| | IRCoT (matched budget) | 42.4 | 42.2 | 21.0 | 20.07 |
| | Adaptive-RAG (matched budget) | 37.0 | 37.0 | 12.8 | 19.42 |
| | Search-o1 | 40.5 | 45.1 | 17.3 | 14.63 |
| | **ReliOpt** | **45.1** | **56.1** | **23.9** | 19.46 |
| Qwen3-8B | Standard RAG | 38.8 | 33.4 | 11.6 | 1.00 |
| | IRCoT (matched budget) | 36.8 | 42.2 | 21.4 | 18.48 |
| | Adaptive-RAG (matched budget) | 39.0 | 35.8 | 19.6 | 18.60 |
| | Search-o1 | 46.5 | 56.1 | 21.8 | 8.75 |
| | **ReliOpt** | **47.9** | **57.9** | **25.2** | 18.62 |

Cost: tokens per question relative to Standard RAG on the same backbone, averaged over the datasets. ReliOpt runs at a
total budget of half the tokens of official IRCoT; IRCoT and Adaptive-RAG are shown at a budget matched to it. Their
official settings, the other baselines, F1, the controls and the quality-cost curves are in the paper.

## Repository layout

```
reliopt/
  plan/planner.py      the planner: question -> typed JSON plan, validation, one retry, fallback
  exec/steps.py        TXT / CMP / LOG execution and the shared pool of sampled re-executions
  exec/operators.py    verification operators: PROBE, VOTE-k, REGROUND, DEEPGROUND, REROUTE
  exec/adaptive.py     priced step-level verification and the Question-Level allocator
  exec/replay.py       one materialisation per question, offline replay of verification policies
  exec/sandbox.py      restricted evaluator for CMP programs
  quality/             step features, step labels from intermediate annotations, the quality model
  baselines/           CoT / Standard RAG / Self-Consistency, Self-Ask, IRCoT, Search-o1, PyRAG, Plan-and-Budget, CoVe
  retrieval/bm25.py    BM25 over IRCoT's corpora (bm25s)
  llm/client.py        client for local vLLM servers; every call is cached on disk by its full prompt
  data/, eval/         dataset loaders; EM and F1 as in IRCoT's evaluation code
scripts/
  e0_*                 data, question ids and BM25 indexes
  e1_*                 pilot runs and the choice of the TXT reader on the tuning questions
  e2_*                 external baselines and their summaries
  e3_*                 the method: materialise calibration questions, fit, price, run, summarise, paired bootstrap
  e8_*                 execution-design ablation of the typed plan
  e9_*                 escalation: development calibration runs, pricing, Adaptive-RAG (labels, T5 classifier, routing)
  make_*               tables and figures from the stored outputs
run/                   shell entry points for e0, e2, e3 and e8 (resumable)
tests/                 a fake vLLM server and toy end-to-end tests that need no GPU and no data
```

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Models (Hugging Face ids): `meta-llama/Llama-3.1-8B-Instruct` and `Qwen/Qwen3-8B` (non-thinking mode) play every role;
`google-t5/t5-large` is the base of the Adaptive-RAG classifier (place it in `cache/hf_models/t5-large/`). To serve a
local copy of the weights, set `MODEL_DIR`.

## Data

Datasets and corpora are not redistributed. Put the following files in `cache/ircot/downloads/` (the sizes and sha256
sums are those we used):

| File | Source | sha256 |
|---|---|---|
| `processed_data.zip` | IRCoT's processed data (`download/processed_data.sh` in the IRCoT repository) | `271fff07efb120a71739c89ab69ab10f4c00059e74b2f7b451a607158c364906` |
| `musique_v1.0.zip` | MuSiQue v1.0 (`download/raw_data.sh` in the IRCoT repository) | `98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd` |
| `2wikimultihopqa_data_ids.zip` | https://www.dropbox.com/s/7ep3h8unu2njfxv/data_ids.zip?dl=1 | `664013d34f169ed5223aedde0c8f013ac6be9568426966319240f722867c5b2d` |
| `hotpotqa_wiki_abstracts.tar.bz2` | https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 | `1acca1c5cc93c4890ea51091d2bad7c3ef6987aead127ab88728dc9e26555729` |

IRCoT's prompts are read from a checkout of https://github.com/StonyBrookNLP/ircot (commit `3c1820f`) in
`cache/ircot/ircot_repo/` (or set `IRCOT_REPO`). Then build the corpora, question ids and indexes:

```bash
bash run/e0_data.sh
```

The test questions are IRCoT's 500-question test subsets and the tuning questions its 100 development questions; 500
training questions per dataset calibrate the quality model, and 1,000 further development questions, disjoint from the
test and tuning questions, calibrate escalation (`scripts/e9_devcal.py --make-ids`).

## Serving the models

```bash
bash run/e0_serve.sh llama31-8b 0,1,2,3      # one vLLM server per GPU on ports 8100 + GPU id; "2+3" = tensor parallel
```

Every script takes `--gpus` with the same ids and spreads its calls over those servers.

## Running the experiments

For one backbone (`T=llama31-8b` or `T=qwen3-8b`, `G` = the GPU ids of its servers):

```bash
# external baselines (IRCoT's K and Plan-and-Budget's retrieval are tuned first) and the Self-Consistency budget B_SC
for ds in hotpotqa 2wikimultihopqa musique; do bash run/e2_baselines.sh $T $ds $G; done
python scripts/e2_summarize.py --tag $T
# the TXT reader, chosen on the tuning questions among three configurations -> results/e3/$T/config.json
python scripts/e1_txt_tune.py --tag $T --gpus $G --configs span:5 span:8 span:8:q --write-config
# the method with step-level verification and the internal comparisons, then paired bootstrap
bash run/e3_method.sh $T $G
python scripts/e3_bootstrap.py --tag $T && python scripts/e3_bootstrap_internal.py --tag $T
# execution-design ablation of the typed plan
bash run/e8_exec_ablation.sh $T $G
```

Escalation (K = IRCoT's tuned K per dataset: 6 / 8 / 6 on Llama-3.1-8B-Instruct, 6 / 6 / 4 on Qwen3-8B):

```bash
python scripts/e9_devcal.py --make-ids && python scripts/e9_devcal.py --make-ids --split devcal2
for s in devcal devcal2; do
  python scripts/e9_devcal.py --tag $T --gpus $G --split $s            # plan and default execution
  for ds in hotpotqa 2wikimultihopqa musique; do
    python scripts/e2_baselines.py --tag $T --gpus $G --split $s --datasets $ds --methods ircot searcho1 \
      --seeds 17 29 43 --ircot-k $K
  done
done
python scripts/e9_price.py --tag $T --calsrc dev1000
```

Adaptive-RAG (silver labels from the 500 training calibration questions, a T5-large classifier per label set):

```bash
for ds in hotpotqa 2wikimultihopqa musique; do
  python scripts/e2_baselines.py --tag $T --gpus $G --split cal --datasets $ds --methods ircot simple searcho1 \
    --seeds 17 --ircot-k $K
done
python scripts/e9_arag_labels.py --tag $T
python scripts/e9_arag_train.py --tag $T --labels abc            # Adaptive-RAG
python scripts/e9_arag_train.py --tag $T --labels plan_ircot     # the pre-execution router over the plan and IRCoT
python scripts/e9_arag_eval.py --tag $T
```

Tables and figures, once both backbones are done (written to `results/tables/` and `results/figures/`):

```bash
mkdir -p results/tables results/figures
python scripts/e9_arag_matched.py && python scripts/e9_signal_check.py && python scripts/e9_escalated.py
python scripts/e9_complementarity.py
python scripts/make_table1.py && python scripts/make_table_esc.py && python scripts/make_tables23.py && python scripts/make_table_e8.py
python scripts/make_table_escalated.py && python scripts/make_table_calsrc.py && python scripts/make_table_rho.py
python scripts/make_table_case.py --tag qwen3-8b --ds musique --qid 3hop1__63037_567566_84283
for t in llama31-8b qwen3-8b; do python scripts/make_fig_qc.py --tag $t; done; python scripts/make_fig3.py
```

All runners are resumable: finished output files are skipped, and unfinished ones continue from the LLM call cache in
`cache/llm/`. The local tests need neither a GPU nor data: `bash tests/toy_e1.sh` and `bash tests/toy_e8.sh`.

## Stored outputs and LLM call caches

Release [v1.0](https://github.com/AlanBjj/ReliOpt/releases/tag/v1.0) holds the outputs of our runs:

| File | Content |
|---|---|
| `results.tar.gz` (53 MB) | every per-question output, the fitted quality models, prices and summaries (`results/e0` to `results/e9`) |
| `llm_cache_llama31-8b.sqlite.gz` (241 MB) | LLM call cache of Llama-3.1-8B-Instruct (full prompt and settings -> response) |
| `llm_cache_qwen3-8b.part{1,2,3}.sqlite.gz` (84 / 19 / 55 MB) | LLM call cache of Qwen3-8B, in three parts |

Extracting `results.tar.gz` in the repository root is enough to regenerate every table and figure with the last block
of commands above, without a GPU. To reuse the caches, decompress them into `cache/llm/` and merge the Qwen3-8B parts:

```bash
tar -xzf results.tar.gz
mkdir -p cache/llm && gunzip -c llm_cache_llama31-8b.sqlite.gz > cache/llm/llama31-8b.sqlite
for i in 1 2 3; do gunzip -c llm_cache_qwen3-8b.part$i.sqlite.gz > cache/llm/qwen3-8b.part$i.sqlite; done
python - <<'EOF'
import sqlite3
con = sqlite3.connect("cache/llm/qwen3-8b.sqlite")
con.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
for i in (1, 2, 3):
    con.execute(f"ATTACH 'cache/llm/qwen3-8b.part{i}.sqlite' AS p")
    con.execute("INSERT OR IGNORE INTO c SELECT * FROM p.c")
    con.commit()
    con.execute("DETACH p")
EOF
```

## License

MIT (see `LICENSE`). The baselines follow the official code and prompts of their authors; see the header of each file in
`reliopt/baselines/` for the source and commit.
