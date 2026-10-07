"""Adaptive-RAG baseline, step 2: fine-tune T5-large as the pre-execution router, then label the 1,500 test questions.

Why: Adaptive-RAG (Jeong et al., NAACL 2024) is the closest competitor of our reliability-based escalation: it decides the
strategy before execution from the question alone. This script reproduces its classifier recipe on our calibration
questions (official code github.com/starsuzi/Adaptive-RAG commit 0c88670, classifier/run_classifier.py, utils.py and
classifier/run/run_large_train_xl.sh):
  model      t5-large (original T5, AutoModelForSeq2SeqLM), full fine-tuning, fp32, dropout as in the config (0.1)
  input      the question text, stripped, no task prefix, truncated at 384 tokens, dynamic padding per batch
  target     the label word as a one-token text target ("A" / "B" / "C" + </s>), cross-entropy loss of the seq2seq model
  optimiser  AdamW, lr 3e-5, weight decay 0, betas/eps torch defaults, linear decay to 0 with no warm-up, batch 32,
             shuffled each epoch, last partial batch kept
  epochs     25 (the official script sweeps 15-35 and picks the epoch on the TEST set's silver labels; their released
             post-processing uses epoch 25 for FLAN-T5-XL; we fix 25 and never look at test labels)
  predict    one decoder step from the start token; softmax over the logits of the label-word tokens; argmax (the code
             takes generate(...).scores[0], which is the same first-step logits)
The ablation label set (plan_ircot: CHEAP / EXPENSIVE) uses the label words "A" / "B" with the same recipe.
Deviations, all forced by our setting: our three multi-hop datasets only (the official classifier also trains on NQ,
TriviaQA, SQuAD), one model per backbone and label set, labels from our backbones' runs instead of FLAN-T5/GPT-3.5,
plain PyTorch loop instead of accelerate (same single-device maths), fixed seed 17 (official: none), CPU by default
(--device cuda:N on a free GPU). Speed on CPU (2026-10-07, 32 threads, Xeon Gold 6246R / Platinum 8352V): about
10 s per batch with the official random batches (padding to the longest question in the batch dominates), about 3.5 s
with --group-by-length (off by default; batches of similar-length questions, batch order shuffled; same loss, only the
batch composition changes), i.e. about 3.3 h vs 70 min for the 1,175 steps of one model; 64 threads were slower than 32.
Input:  results/e9/<tag>/arag/labels_<set>.jsonl, test_questions.jsonl (scripts/e9_arag_labels.py); t5-large weights
        (server: cache/hf_models/t5-large, google-t5/t5-large revision 150ebc2, see cache/SOURCES.md)
Output: results/e9/<tag>/arag/pred_<set>.jsonl   {qid, ds, pred, probs: {label: p}} for the 1,500 test questions
        results/e9/<tag>/arag/train_<set>.json   recipe, per-epoch loss and time, accuracy on the training questions
Usage:  OMP_NUM_THREADS=32 python scripts/e9_arag_train.py --tag llama31-8b --labels abc [--threads 32]
        [--group-by-length] [--device cuda:0]
        dry run: ... --limit 8 --epochs 1 --data-dir <toy labels dir> --out-dir logs/e9_arag_dry/llama31-8b
Resumable: model + optimiser + scheduler + RNG state are saved to <out-dir>/ckpt_<set>/state.pt every --ckpt-every epochs
(about 9 GB, server only, never copy it into iCloud); a rerun of the same command continues from there. The checkpoint
directory is deleted once pred_<set>.jsonl is written (pass --keep-ckpt to keep it).
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
WORDS = {"abc": {"A": "A", "B": "B", "C": "C"}, "plan_ircot": {"CHEAP": "A", "EXPENSIVE": "B"}}   # label -> target word


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def first_per_ds(rows, n):
    if not n:
        return rows
    out, seen = [], {}
    for r in rows:
        if seen.get(r["ds"], 0) < n:
            out.append(r)
            seen[r["ds"]] = seen.get(r["ds"], 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--labels", required=True, choices=sorted(WORDS))
    ap.add_argument("--model", default="cache/hf_models/t5-large")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=100)
    ap.add_argument("--max-seq-length", type=int, default=384)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--device", default="cpu", help="cpu (default) or e.g. cuda:0 when a free GPU is approved")
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--keep-ckpt", action="store_true")
    ap.add_argument("--group-by-length", action="store_true",
                    help="CPU speed-up, off by default (the official code shuffles without length grouping): batches "
                         "of similar-length questions, as transformers' group_by_length")
    ap.add_argument("--limit", type=int, default=0, help="dry run: first N questions per dataset (train and test)")
    ap.add_argument("--out-dir", default=None, help="default results/e9/<tag>/arag")
    ap.add_argument("--data-dir", default=None, help="where labels_<set>.jsonl / test_questions.jsonl are; default "
                                                    "results/e9/<tag>/arag (dry runs point it at toy labels)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_scheduler
    torch.set_num_threads(args.threads)

    data_dir = ROOT / args.data_dir if args.data_dir else ROOT / "results" / "e9" / args.tag / "arag"
    out_dir = ROOT / args.out_dir if args.out_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_p, log_p = out_dir / f"pred_{args.labels}.jsonl", out_dir / f"train_{args.labels}.json"
    if pred_p.exists():
        print(f"skip: {pred_p} exists"); return
    ckpt_dir = out_dir / f"ckpt_{args.labels}"
    ckpt_p = ckpt_dir / "state.pt"

    train = first_per_ds(read_jsonl(data_dir / f"labels_{args.labels}.jsonl"), args.limit)
    test = first_per_ds(read_jsonl(data_dir / "test_questions.jsonl"), args.limit)
    words = WORDS[args.labels]
    classes = list(words)
    assert {r["label"] for r in train} <= set(classes), {r["label"] for r in train}
    print(f"{args.tag} {args.labels}: {len(train)} training questions, {len(test)} test questions, threads {args.threads}",
          flush=True)

    model_dir = ROOT / args.model if not os.path.isabs(args.model) else Path(args.model)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(args.device)
    word_ids = []
    for c in classes:
        ids = tok(words[c]).input_ids
        assert len(ids) == 2 and ids[1] == tok.eos_token_id, (c, ids)   # one piece + </s>, as in the official code
        word_ids.append(ids[0])
    assert len(set(word_ids)) == len(word_ids)

    def encode(rows):
        enc = tok([r["question"].strip() for r in rows], truncation=True, max_length=args.max_seq_length)
        return enc["input_ids"]

    def pad(seqs, value):
        n = max(len(s) for s in seqs)
        return torch.tensor([s + [value] * (n - len(s)) for s in seqs])

    x_train, x_test = encode(train), encode(test)
    n_trunc = sum(len(s) >= args.max_seq_length for s in x_train + x_test)
    y_train = [tok(text_target=words[r["label"]], max_length=30, truncation=True).input_ids for r in train]
    lens = sorted(len(s) for s in x_train)
    print(f"question tokens: median {lens[len(lens) // 2]}, max {lens[-1]}; truncated {n_trunc}", flush=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # official grouping (no decay for bias / layer norm) is moot with weight decay 0
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    sched = get_scheduler("linear", optimizer=opt, num_warmup_steps=0, num_training_steps=total_steps)
    start_epoch, history = 0, []
    if ckpt_p.exists():
        st = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["rng"]); random.setstate(st["py_rng"])
        start_epoch, history = st["epoch"], st["history"]
        print(f"resumed from {ckpt_p} after epoch {start_epoch}", flush=True)

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, total = time.time(), 0.0
        perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed * 1000 + epoch)).tolist()
        if args.group_by_length:   # transformers' LengthGroupedSampler: sort inside mega-batches of 50 batches,
            mb = 50 * args.batch_size                        # then shuffle the order of the batches
            perm = [i for k in range(0, len(perm), mb)
                    for i in sorted(perm[k:k + mb], key=lambda i: -len(x_train[i]))]
        batches = [perm[k:k + args.batch_size] for k in range(0, len(perm), args.batch_size)]
        if args.group_by_length:
            order = torch.randperm(len(batches), generator=torch.Generator().manual_seed(args.seed * 1000 + epoch + 1))
            batches = [batches[c] for c in order.tolist()]
        assert len(batches) == steps_per_epoch
        for b, idx in enumerate(batches):
            ids = pad([x_train[i] for i in idx], tok.pad_token_id).to(args.device)
            labels = pad([y_train[i] for i in idx], -100).to(args.device)
            loss = model(input_ids=ids, attention_mask=(ids != tok.pad_token_id).long(), labels=labels).loss
            loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            total += loss.item()
            if epoch == start_epoch and b < 3:
                print(f"  step {b + 1}/{steps_per_epoch}: {time.time() - t0:.1f} s, loss {loss.item():.4f}", flush=True)
        history.append({"epoch": epoch + 1, "loss": total / steps_per_epoch, "seconds": round(time.time() - t0, 1)})
        print(f"epoch {epoch + 1}/{args.epochs}: loss {history[-1]['loss']:.4f}, {history[-1]['seconds']} s", flush=True)
        if (epoch + 1) % args.ckpt_every == 0 or epoch + 1 == args.epochs:
            ckpt_dir.mkdir(exist_ok=True)
            tmp = ckpt_p.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": torch.get_rng_state(), "py_rng": random.getstate(), "epoch": epoch + 1,
                        "history": history}, tmp)
            os.replace(tmp, ckpt_p)

    @torch.no_grad()
    def predict(xs):
        model.eval()
        out = []
        for b in range(0, len(xs), args.eval_batch_size):
            ids = pad(xs[b:b + args.eval_batch_size], tok.pad_token_id).to(args.device)
            dec = torch.full((ids.shape[0], 1), model.config.decoder_start_token_id, dtype=torch.long, device=args.device)
            logits = model(input_ids=ids, attention_mask=(ids != tok.pad_token_id).long(), decoder_input_ids=dec).logits
            out += torch.softmax(logits[:, 0, word_ids], dim=-1).tolist()
        return out

    t0 = time.time()
    p_test = predict(x_test)
    p_train = predict(x_train)
    pred_seconds = round(time.time() - t0, 1)
    train_acc = sum(classes[max(range(len(classes)), key=p.__getitem__)] == r["label"]
                    for p, r in zip(p_train, train)) / len(train)
    rows = [{"qid": r["qid"], "ds": r["ds"], "pred": classes[max(range(len(classes)), key=p.__getitem__)],
             "probs": {c: round(v, 6) for c, v in zip(classes, p)}} for r, p in zip(test, p_test)]
    log = {"tag": args.tag, "labels": args.labels, "model": str(args.model), "words": words, "word_ids": word_ids,
           "n_train": len(train), "n_test": len(test), "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
           "steps": total_steps, "max_seq_length": args.max_seq_length, "seed": args.seed, "threads": args.threads,
           "truncated": n_trunc, "history": history, "train_acc": train_acc,
           "train_seconds_this_run": round(time.time() - t_start - pred_seconds, 1),
           "train_seconds_total": round(sum(h["seconds"] for h in history), 1), "predict_seconds": pred_seconds,
           "group_by_length": args.group_by_length, "device": args.device, "limit": args.limit}
    log_p.write_text(json.dumps(log, indent=1) + "\n")
    tmp = pred_p.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(x) + "\n" for x in rows))
    os.replace(tmp, pred_p)
    print(f"wrote {pred_p} ({len(rows)} rows); training accuracy {train_acc:.3f}; {log_p}", flush=True)
    if ckpt_dir.exists() and not args.keep_ckpt:
        shutil.rmtree(ckpt_dir)


if __name__ == "__main__":
    sys.exit(main())
