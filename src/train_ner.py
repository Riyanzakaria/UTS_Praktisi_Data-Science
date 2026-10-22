"""
train_ner.py — Fine-tune the NER component of en_core_web_sm on the 12 train
records of project/data/annotations.jsonl (NER coursework study case).

Procedure:
  1. Load the gold annotations; split into train / val / test by the "split" field.
  2. Fix all randomness (spaCy + random + numpy, seed 42).
  3. Fine-tune the NER pipe of en_core_web_sm (~25 iterations, dropout 0.2,
     compounding minibatch sizes, spaCy Example objects).
  4. After each iteration, score exact-span F1 on the 3 val records.
     Keep the iteration with the best val F1 (greedy checkpointing in memory).
  5. Save the best pipeline to project/models/ner_model.
  6. Print + record settings, duration, doc/span counts per split, imbalance note.

NOTE: test records are NEVER used here — not for training, not for tuning.
The held-out evaluation (val + 10-item set) is done separately in evaluate.py.
"""
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import spacy
from spacy.training import Example
from spacy.util import compounding, minibatch

# ------------------------------------------------------------------ config
SEED = 42
ITERATIONS = 25
DROPOUT = 0.2
MIN_BATCH, MAX_BATCH = 4.0, 32.0
LABELS = ["PERSON", "ORG", "GPE", "DATE", "EVENT", "MONEY"]

PROJECT = Path(__file__).resolve().parent.parent          # project/
DATA = PROJECT / "data" / "annotations.jsonl"
MODEL_DIR = PROJECT / "models" / "ner_model"
LOG = PROJECT / "data" / "training_log.json"


def load_records(path: Path):
    """Read gold JSONL; each record carries 'split' and 'entities' as
    (start, end, label) char offsets into 'text'."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_examples(nlp, records):
    """Build spaCy Example objects from (text, {'entities': [(s,e,label)]}).

    The CSV 'entities' column is deliberately ignored (it is empty anyway);
    the JSONL gold spans are the only labels used."""
    examples = []
    for rec in records:
        ann = {"entities": [(e["start"], e["end"], e["label"]) for e in rec["entities"]]}
        doc = nlp.make_doc(rec["text"])
        examples.append(Example.from_dict(doc, ann))
    return examples


def exact_f1(nlp, records):
    """Exact-span F1: a prediction counts iff (start, end, label) == gold span."""
    tp = fp = fn = 0
    for rec in records:
        gold = {(d["start"], d["end"], d["label"]) for d in rec["entities"]}
        pred = {(e.start_char, e.end_char, e.label_) for e in nlp(rec["text"]).ents}
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main():
    t0 = time.time()

    # --- 1. data -----------------------------------------------------------
    records = load_records(DATA)
    train_recs = [r for r in records if r["split"] == "train"]
    val_recs = [r for r in records if r["split"] == "val"]
    test_recs = [r for r in records if r["split"] == "test"]
    assert len(train_recs) == 12 and len(val_recs) == 3 and len(test_recs) == 3, \
        f"unexpected split sizes: {len(train_recs)}/{len(val_recs)}/{len(test_recs)}"

    span_counts = {s: Counter() for s in ("train", "val", "test")}
    for rec in records:
        for d in rec["entities"]:
            span_counts[rec["split"]][d["label"]] += 1

    # --- 2. reproducibility --------------------------------------------------
    spacy.util.fix_random_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # --- 3. fine-tune NER ----------------------------------------------------
    nlp = spacy.load("en_core_web_sm")
    ner = nlp.get_pipe("ner")
    for lab in LABELS:          # no-op adds for existing labels; safe anyway
        ner.add_label(lab)
    other_pipes = [p for p in nlp.pipe_names if p != "ner"]

    examples = to_examples(nlp, train_recs)
    optimizer = nlp.resume_training()  # keeps pretrained weights, restarts optimizer

    best_f1, best_bytes, val_history = -1.0, None, []
    with nlp.disable_pipes(*other_pipes):
        for it in range(1, ITERATIONS + 1):
            random.shuffle(examples)          # seeded shuffle
            batches = minibatch(examples, size=compounding(MIN_BATCH, MAX_BATCH, 1.001))
            for batch in batches:
                nlp.update(batch, sgd=optimizer, drop=DROPOUT)
            p, r, f1 = exact_f1(nlp, val_recs)  # checkpoint on val only
            val_history.append({"iteration": it, "val_precision": p,
                                "val_recall": r, "val_f1": f1})
            print(f"iter {it:2d}/{ITERATIONS}  val P={p:.3f} R={r:.3f} F1={f1:.3f}", flush=True)
            if f1 > best_f1:                  # strict '>' → ties keep earliest iter
                best_f1 = f1
                best_bytes = nlp.to_bytes()

    duration = time.time() - t0
    best_iter = max(val_history, key=lambda h: (h["val_f1"], -h["iteration"]))

    # --- 4. save best pipeline -----------------------------------------------
    nlp.from_bytes(best_bytes)   # restore the weights of the best iteration
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_DIR.exists():
        import shutil
        shutil.rmtree(MODEL_DIR)
    nlp.to_disk(MODEL_DIR)

    # --- 5. log + report -------------------------------------------------------
    doc_spans = {
        s: {"docs": len([r for r in records if r["split"] == s]),
            "spans": sum(span_counts[s].values()),
            "per_label": dict(span_counts[s])}
        for s in ("train", "val", "test")
    }
    train_spans = sum(span_counts["train"].values())
    imbalance = (
        f"Class imbalance (train): top labels = "
        f"{', '.join(f'{k}:{v}' for k, v in span_counts['train'].most_common(3))}; "
        f"rarest = {', '.join(f'{k}:{v}' for k, v in span_counts['train'].most_common()[-2:])}. "
        f"Model trains on 12 docs / {train_spans} spans."
    )
    log = {
        "seed": SEED, "iterations": ITERATIONS, "dropout": DROPOUT,
        "batch_compounding": [MIN_BATCH, MAX_BATCH], "labels": LABELS,
        "duration_seconds": round(duration, 1),
        "doc_span_counts": doc_spans,
        "imbalance_note": imbalance,
        "val_history": val_history,
        "best_iteration": best_iter["iteration"],
        "best_val_f1": best_iter["val_f1"],
        "stopping_choice": ("After each iteration scored exact-span F1 on the 3 val "
                            "records; kept the single iteration with the highest val F1 "
                            f"(iteration {best_iter['iteration']}, F1={best_iter['val_f1']:.3f}; "
                            "ties resolved to the earliest iteration). Test records were "
                            "never used during training or tuning."),
        "held_out_note": ("The 3 test records of annotations.jsonl and the 7 new "
                          "eval_extra excerpts are HELD OUT: they were not used in "
                          "training and did not influence model or iteration selection."),
    }
    LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"duration: {duration:.1f}s | best iter: {best_iter['iteration']} "
          f"(val F1={best_iter['val_f1']:.3f}) | model saved to {MODEL_DIR}")
    print(f"docs/spans — train: 12/{doc_spans['train']['spans']}, "
          f"val: 3/{doc_spans['val']['spans']}, test: 3/{doc_spans['test']['spans']}")
    print(imbalance)


if __name__ == "__main__":
    main()
