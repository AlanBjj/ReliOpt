"""e0: unpack the IRCoT release and raw data, build the three retrieval corpora, lock question IDs.

Input:  cache/ircot/downloads/{processed_data.zip, musique_v1.0.zip, 2wikimultihopqa_data_ids.zip,
                               hotpotqa_wiki_abstracts.tar.bz2}
Output: cache/ircot/processed_data/<ds>/*.jsonl, cache/ircot/raw_data/<ds>/..., cache/ircot/corpus/<ds>.jsonl (server only),
        results/e0/data_manifest.json, results/e0/ids/<ds>_{test,tune,cal}.txt (small, pulled back)
Usage:  python scripts/e0_prepare_data.py [--workers 32]
Resumable: every stage checks for its finished output and skips; corpora are written to *.partial then renamed.

Corpora follow IRCoT's retriever_server/build_index.py: HotpotQA = the 2017 Wikipedia abstracts shipped with HotpotQA;
2WikiMultiHopQA and MuSiQue = all paragraphs of their train/dev/test files, de-duplicated on title + text.
Calibration questions: 500 per dataset drawn with a fixed seed from IRCoT's processed train split; tuning questions are
IRCoT's 100-question dev subset; test questions are IRCoT's 500-question test subset.
"""
import argparse
import bz2
import glob
import hashlib
import json
import random
import shutil
import sys
import tarfile
import zipfile
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reliopt.data.ircot import DATASETS, IRCOT_DIR, load_questions  # noqa: E402

DL = IRCOT_DIR / "downloads"
RAW = IRCOT_DIR / "raw_data"
CORPUS = IRCOT_DIR / "corpus"
OUT = ROOT / "results" / "e0"
CAL_SEED, CAL_N = 20261007, 500
IRCOT_CORPUS_SIZES = {"hotpotqa": 5233329, "2wikimultihopqa": 430225, "musique": 139416}   # IRCoT README


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def unzip_flat(zpath, dest):
    """Extract every file of a zip into dest, dropping directory components (like `unzip -j`)."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            if info.is_dir() or not name or name.startswith(".") or "__MACOSX" in info.filename:
                continue
            with z.open(info) as src, open(dest / name, "wb") as dst:
                shutil.copyfileobj(src, dst)


def stage_unpack():
    if not (IRCOT_DIR / "processed_data" / "musique" / "test_subsampled.jsonl").exists():
        print("unpack processed_data.zip"); sys.stdout.flush()
        with zipfile.ZipFile(DL / "processed_data.zip") as z:
            z.extractall(IRCOT_DIR, members=[m for m in z.namelist() if ".DS_Store" not in m and "__MACOSX" not in m])
    if not (RAW / "2wikimultihopqa" / "dev.json").exists():
        print("unpack 2wikimultihopqa"); sys.stdout.flush()
        unzip_flat(DL / "2wikimultihopqa_data_ids.zip", RAW / "2wikimultihopqa")
    if not (RAW / "musique" / "musique_ans_v1.0_dev.jsonl").exists():
        print("unpack musique"); sys.stdout.flush()
        unzip_flat(DL / "musique_v1.0.zip", RAW / "musique")
    target = RAW / "hotpotqa" / "wikipedia-paragraphs"
    if not target.exists():
        print("unpack hotpotqa abstracts (several minutes)"); sys.stdout.flush()
        tmp = RAW / "hotpotqa" / "_extract"
        tmp.mkdir(parents=True, exist_ok=True)
        with tarfile.open(DL / "hotpotqa_wiki_abstracts.tar.bz2", "r:bz2") as t:
            try:
                t.extractall(tmp, filter="data")
            except TypeError:   # Python < 3.12
                t.extractall(tmp)
        tops = [p for p in tmp.iterdir() if p.is_dir()]
        assert len(tops) == 1, tops
        tops[0].rename(target)
        tmp.rmdir()


def _hotpot_file(path):
    rows = []
    with bz2.open(path, "rt") as f:
        for line in f:
            inst = json.loads(line)
            parts = []
            for e in inst.get("text", []):
                parts += [x.strip() for x in e] if isinstance(e, list) else [str(e).strip()]
            rows.append((inst.get("title", ""), " ".join(p for p in parts if p)))
    return rows


def _write_corpus(ds, docs_iter):
    out = CORPUS / f"{ds}.jsonl"
    part = CORPUS / f"{ds}.jsonl.partial"
    n = 0
    with open(part, "w") as f:
        for title, text in docs_iter:
            f.write(json.dumps({"id": f"{ds}-{n}", "title": title, "text": text}, ensure_ascii=False) + "\n")
            n += 1
    part.rename(out)
    return n


def _dedup(pairs):
    seen = set()
    for title, text in pairs:
        key = hashlib.md5((title + " " + text).encode()).digest()
        if key in seen:
            continue
        seen.add(key)
        yield title, text


def build_corpus(ds, workers):
    out = CORPUS / f"{ds}.jsonl"
    if out.exists():
        return sum(1 for _ in open(out))
    CORPUS.mkdir(parents=True, exist_ok=True)
    print(f"build corpus {ds}"); sys.stdout.flush()
    if ds == "hotpotqa":
        files = sorted(glob.glob(str(RAW / "hotpotqa" / "wikipedia-paragraphs" / "*" / "wiki_*.bz2")))
        assert files, "no HotpotQA abstract files"

        def it():
            with Pool(workers) as pool:
                for rows in pool.imap(_hotpot_file, files, chunksize=4):
                    yield from rows
        return _write_corpus(ds, it())
    if ds == "2wikimultihopqa":
        def it():
            for split in ("train", "dev", "test"):
                for item in json.load(open(RAW / ds / f"{split}.json")):
                    for title, sents in item["context"]:
                        yield title, " ".join(sents)
        return _write_corpus(ds, _dedup(it()))
    def it():
        for path in sorted(glob.glob(str(RAW / "musique" / "*.jsonl"))):
            for line in open(path):
                for p in json.loads(line)["paragraphs"]:
                    yield p["title"], p["paragraph_text"]
    return _write_corpus(ds, _dedup(it()))


def write_ids(ds):
    ids_dir = OUT / "ids"
    ids_dir.mkdir(parents=True, exist_ok=True)
    test = load_questions(ds, "test_subsampled")
    tune = load_questions(ds, "dev_subsampled")
    train_ids = sorted(json.loads(l)["question_id"] for l in open(IRCOT_DIR / "processed_data" / ds / "train.jsonl"))
    cal = sorted(random.Random(CAL_SEED).sample(train_ids, CAL_N))
    assert not (set(cal) & {q["qid"] for q in test}), "calibration overlaps test"
    for name, qids in (("test", [q["qid"] for q in test]), ("tune", [q["qid"] for q in tune]), ("cal", cal)):
        path = ids_dir / f"{ds}_{name}.txt"
        if not path.exists():
            path.write_text("\n".join(qids) + "\n")
    types = {}
    for q in test:
        types[q["qtype"]] = types.get(q["qtype"], 0) + 1
    return {"n_test": len(test), "n_tune": len(tune), "n_train": len(train_ids), "n_cal": len(cal),
            "test_types": dict(sorted(types.items())),
            "test_with_steps": sum(1 for q in test if q["steps"]),
            "test_answers_empty": sum(1 for q in test if not q["answers"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stage_unpack()
    manifest = {"downloads": {}, "datasets": {}, "cal_seed": CAL_SEED}
    for p in sorted(DL.glob("*")):
        if p.is_file():
            manifest["downloads"][p.name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    for ds in DATASETS:
        n_docs = build_corpus(ds, args.workers)
        info = write_ids(ds)
        info["corpus_docs"] = n_docs
        info["corpus_docs_ircot"] = IRCOT_CORPUS_SIZES[ds]
        for split in ("test_subsampled", "dev_subsampled"):
            info[f"sha256_{split}"] = sha256(IRCOT_DIR / "processed_data" / ds / f"{split}.jsonl")
        manifest["datasets"][ds] = info
        print(ds, json.dumps(info)); sys.stdout.flush()
    (OUT / "data_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("wrote", OUT / "data_manifest.json")


if __name__ == "__main__":
    main()
