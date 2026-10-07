"""BM25 retrieval over the per-dataset corpora (bm25s, Lucene variant with Elasticsearch defaults k1=1.2, b=0.75).

IRCoT indexes title and paragraph text in Elasticsearch; here the two are concatenated into one field. Index files live in
cache/bm25/<ds>/ (built by scripts/e0_build_bm25.py) next to the corpus in cache/ircot/corpus/<ds>.jsonl.
"""
import json

from ..utils.env import ROOT

BM25_DIR = ROOT / "cache" / "bm25"
CORPUS_DIR = ROOT / "cache" / "ircot" / "corpus"
K1, B = 1.2, 0.75


def doc_text(title, text):
    return f"{title}. {text}" if title else text


def _stemmer():
    import Stemmer
    return Stemmer.Stemmer("english")


class BM25Index:
    """search(queries, k) -> per query a list of {"id", "title", "text", "score"}; thread-safe."""

    def __init__(self, ds, mmap=True):
        import bm25s
        self.ds = ds
        self.retriever = bm25s.BM25.load(str(BM25_DIR / ds), mmap=mmap)
        self.stemmer = _stemmer()
        self.ids, self.titles, self.texts = [], [], []
        with open(CORPUS_DIR / f"{ds}.jsonl") as f:
            for line in f:
                d = json.loads(line)
                self.ids.append(d["id"]); self.titles.append(d["title"]); self.texts.append(d["text"])
        self._cache = {}

    def search(self, queries, k=5):
        import bm25s
        single = isinstance(queries, str)
        queries = [queries] if single else list(queries)
        todo = [q for q in dict.fromkeys(queries) if (q, k) not in self._cache]
        if todo:
            # string tokens (return_ids=False) are mapped onto the index vocabulary by retrieve(); unknown words drop out
            toks = bm25s.tokenize(todo, stopwords="en", stemmer=self.stemmer, return_ids=False, show_progress=False)
            vocab = self.retriever.vocab_dict
            live = [i for i, t in enumerate(toks) if any(w in vocab for w in t)]
            for i in set(range(len(todo))) - set(live):
                self._cache[(todo[i], k)] = []   # nothing searchable in the query
            if live:
                # no lock: retrieval only reads the (memory-mapped) index; with 16 threads on HotpotQA (5.2M passages) it
                # returned the same top-8 lists for 80 test questions as sequential calls, 3.6x faster (2026-10-07)
                idx, scores = self.retriever.retrieve([[w for w in toks[i] if w in vocab] for i in live],
                                                      k=min(k, len(self.ids)), show_progress=False, n_threads=1)
                for i, row, srow in zip(live, idx, scores):
                    self._cache[(todo[i], k)] = [{"id": self.ids[j], "title": self.titles[j], "text": self.texts[j],
                                                  "score": float(s)} for j, s in zip(row, srow) if s > 0]
        out = [self._cache[(q, k)] for q in queries]
        return out[0] if single else out
