"""e0: build one BM25 index per dataset and check retrieval on the test subset.

Input:  cache/ircot/corpus/<ds>.jsonl, cache/ircot/processed_data/<ds>/test_subsampled.jsonl
Output: cache/bm25/<ds>/ (server only); results/e0/bm25_sanity.json (support-title recall of the raw question, small)
Usage:  python scripts/e0_build_bm25.py [--datasets hotpotqa 2wikimultihopqa musique]
Resumable: a dataset whose index directory already holds params.index.json is not rebuilt.

The sanity check only asks whether the index works: with the original question as the query, what fraction of supporting
paragraph titles appear in the top-k. Multi-hop questions are not expected to reach high recall in one shot.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reliopt.data.ircot import DATASETS, load_questions  # noqa: E402
from reliopt.retrieval.bm25 import B, BM25_DIR, CORPUS_DIR, K1, BM25Index, _stemmer, doc_text  # noqa: E402


def build(ds):
    import bm25s
    out = BM25_DIR / ds
    if (out / "params.index.json").exists():
        print(f"skip index {ds}"); return
    t0 = time.time()
    texts = []
    with open(CORPUS_DIR / f"{ds}.jsonl") as f:
        for line in f:
            d = json.loads(line)
            texts.append(doc_text(d["title"], d["text"]))
    print(f"{ds}: {len(texts)} docs loaded in {time.time() - t0:.0f}s"); sys.stdout.flush()
    toks = bm25s.tokenize(texts, stopwords="en", stemmer=_stemmer(), show_progress=True)
    del texts
    retriever = bm25s.BM25(method="lucene", k1=K1, b=B)
    retriever.index(toks, show_progress=True)
    tmp = BM25_DIR / f"{ds}.partial"
    retriever.save(str(tmp))
    tmp.rename(out)
    print(f"{ds}: index built in {time.time() - t0:.0f}s"); sys.stdout.flush()


def sanity(ds):
    idx = BM25Index(ds)
    qs = load_questions(ds, "test_subsampled")
    t0 = time.time()
    hits = idx.search([q["question"] for q in qs], k=10)
    secs = time.time() - t0
    res = {}
    for k in (5, 10):
        rec = []
        for q, h in zip(qs, hits):
            gold = set(q["support_titles"])
            if gold:
                rec.append(len(gold & {d["title"] for d in h[:k]}) / len(gold))
        res[f"support_recall@{k}"] = round(100 * sum(rec) / max(len(rec), 1), 1)
    res.update({"n_questions": len(qs), "seconds_for_all_queries": round(secs, 1), "n_docs": len(idx.ids)})
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = ap.parse_args()
    BM25_DIR.mkdir(parents=True, exist_ok=True)
    out = ROOT / "results" / "e0" / "bm25_sanity.json"
    report = json.loads(out.read_text()) if out.exists() else {}
    for ds in args.datasets:
        build(ds)
        report[ds] = sanity(ds)
        print(ds, json.dumps(report[ds])); sys.stdout.flush()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
