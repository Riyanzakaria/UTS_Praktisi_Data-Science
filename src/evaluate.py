"""
evaluate.py — Evaluate baseline (en_core_web_sm) vs fine-tuned NER on HELD-OUT data.

Baseline label mapping (en_core_web_sm -> our scheme), per annotation_guidelines.md:
    PERSON -> PERSON | ORG -> ORG | GPE, LOC -> GPE | DATE, TIME -> DATE
    EVENT -> EVENT | MONEY -> MONEY ; every other label is dropped.

Metric: EXACT-span match. A predicted span is correct iff
(start_char, end_char, label) equals a gold span exactly.

Evaluated on:
  (a) the 3 val records of annotations.jsonl  (used only for iteration
      selection during training, never for weight updates), and
  (b) the 10-item evaluation set (3 test records + 7 new gold excerpts).

The test/eval items never influenced training or tuning in any way.

Writes project/data/evaluation_results.json and prints an error analysis.
"""
import json
from collections import defaultdict
from pathlib import Path

import spacy

# ------------------------------------------------------------------ config
PROJECT = Path(__file__).resolve().parent.parent
ANNOT = PROJECT / "data" / "annotations.jsonl"
EVAL10 = PROJECT / "data" / "evaluation_examples.json"
MODEL_DIR = PROJECT / "models" / "ner_model"
OUT = PROJECT / "data" / "evaluation_results.json"

# Baseline mapping: en_core_web_sm label -> our scheme (None = dropped).
BASELINE_MAP = {
    "PERSON": "PERSON", "ORG": "ORG",
    "GPE": "GPE", "LOC": "GPE",
    "DATE": "DATE", "TIME": "DATE",
    "EVENT": "EVENT", "MONEY": "MONEY",
}


def gold_spans(item):
    """Gold spans from our JSON format: {(start, end, label)}."""
    return {(e["start"], e["end"], e["label"]) for e in item["entities"]}


def pred_spans(nlp, text, mapping=None):
    """Predicted spans; with mapping, translate + drop out-of-scheme labels."""
    spans = set()
    for ent in nlp(text).ents:
        label = mapping.get(ent.label_) if mapping else ent.label_
        if label:
            spans.add((ent.start_char, ent.end_char, label))
    return spans


def score_from_counts(agg, n_gold, n_pred, labels):
    """Build the overall + per-label report from summed tp/fp/fn counts."""
    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                "tp": tp, "fp": fp, "fn": fn}

    out = {"overall": prf(agg["tp"], agg["fp"], agg["fn"]), "per_label": {}}
    out["overall"]["support_gold"] = n_gold
    out["overall"]["support_pred"] = n_pred
    for lab in labels:
        tp, fp, fn = agg["pl"][lab]
        d = prf(tp, fp, fn)
        d["support_gold"] = tp + fn
        d["support_pred"] = tp + fp
        out["per_label"][lab] = d
    return out


def score(gold, pred, labels):
    """Micro overall + per-label precision/recall/F1 on exact-span match."""
    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                "tp": tp, "fp": fp, "fn": fn}

    tp = len(gold & pred)
    out = {"overall": prf(tp, len(pred - gold), len(gold - pred)), "per_label": {}}
    out["overall"]["support_gold"] = len(gold)
    out["overall"]["support_pred"] = len(pred)
    for lab in labels:
        g = {s for s in gold if s[2] == lab}
        pr = {s for s in pred if s[2] == lab}
        m = g & pr
        d = prf(len(m), len(pr - m), len(g - m))
        d["support_gold"] = len(g)
        d["support_pred"] = len(pr)
        out["per_label"][lab] = d
    return out


def error_examples(items, pred_fn, limit=12):
    """Collect (kind, text_snippet, span_text, gold_label, pred_label) cases."""
    cases = []
    for item in items:
        text = item["text"]
        gold = gold_spans(item)
        pred = pred_fn(text)
        for s, e, lab in sorted(pred - gold):
            cases.append(("FP", text[max(0, s - 40):e + 40].replace("\n", " "),
                          text[s:e], None, lab))
        for s, e, lab in sorted(gold - pred):
            cases.append(("FN", text[max(0, s - 40):e + 40].replace("\n", " "),
                          text[s:e], lab, None))
        if len(cases) >= limit:
            break
    return cases


def main():
    labels = ["PERSON", "ORG", "GPE", "DATE", "EVENT", "MONEY"]

    # --- held-out data -------------------------------------------------------
    val_items, test_items = [], []
    with open(ANNOT, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["split"] == "val":
                val_items.append(r)
            elif r["split"] == "test":
                test_items.append(r)
    eval10_items = json.load(open(EVAL10, encoding="utf-8"))
    assert len(val_items) == 3 and len(eval10_items) == 10

    # --- models --------------------------------------------------------------
    baseline = spacy.load("en_core_web_sm")
    trained = spacy.load(MODEL_DIR)
    base_pred = lambda t: pred_spans(baseline, t, BASELINE_MAP)
    train_pred = lambda t: pred_spans(trained, t)

    results = {"val": {}, "eval10": {}}
    for name, items in (("val", val_items), ("eval10", eval10_items)):
        for mname, pfn in (("baseline", base_pred), ("trained", train_pred)):
            # score per item and sum (spans are document-local; a global union
            # would wrongly collapse identical (start,end,label) across docs)
            agg = {"gold": set(), "pred": set(), "tp": 0, "fp": 0, "fn": 0,
                   "pl": defaultdict(lambda: [0, 0, 0])}
            n_gold = n_pred = 0
            for item in items:
                gold = gold_spans(item)
                pred = pfn(item["text"])
                n_gold += len(gold)
                n_pred += len(pred)
                agg["tp"] += len(gold & pred)
                agg["fp"] += len(pred - gold)
                agg["fn"] += len(gold - pred)
                for lab in labels:
                    g = {s for s in gold if s[2] == lab}
                    pr = {s for s in pred if s[2] == lab}
                    m = g & pr
                    cell = agg["pl"][lab]
                    cell[0] += len(m)
                    cell[1] += len(pr - m)
                    cell[2] += len(g - m)
            results[name][mname] = score_from_counts(agg, n_gold, n_pred, labels)
        results[name]["n_items"] = len(items)

    # --- training config (copied from the training log) ----------------------
    log = json.load(open(PROJECT / "data" / "training_log.json", encoding="utf-8"))
    results["training_config"] = {
        "seed": log["seed"], "iterations": log["iterations"],
        "dropout": log["dropout"], "batch_compounding": log["batch_compounding"],
        "labels": log["labels"], "duration_seconds": log["duration_seconds"],
        "best_iteration": log["best_iteration"], "best_val_f1": log["best_val_f1"],
        "baseline_label_mapping": {k: v for k, v in BASELINE_MAP.items()},
        "held_out_statement": (
            "The 3 test records and 7 eval excerpts were NEVER used in training "
            "and did not influence model weights or iteration selection; val F1 "
            "on the 3 val records was the only tuning signal."),
    }
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- console summary -------------------------------------------------------
    for name in ("val", "eval10"):
        print(f"\n### {name} ({results[name]['n_items']} items)")
        for mname in ("baseline", "trained"):
            o = results[name][mname]["overall"]
            print(f"  {mname:8s} P={o['precision']:.3f} R={o['recall']:.3f} "
                  f"F1={o['f1']:.3f}  (gold={o['support_gold']}, pred={o['support_pred']})")
        print("  per-label F1 (baseline -> trained):")
        for lab in labels:
            b = results[name]["baseline"]["per_label"][lab]["f1"]
            t = results[name]["trained"]["per_label"][lab]["f1"]
            print(f"    {lab:6s} {b:.3f} -> {t:.3f}")

    # --- error analysis on the 10-item set ------------------------------------
    print("\n### error examples (eval10, trained model)")
    for kind, ctx, span, glab, plab in error_examples(eval10_items, train_pred):
        detail = (f"gold={glab} predicted nothing" if kind == "FN"
                  else f"predicted {plab}, gold=nothing")
        print(f"  [{kind}] '{span}' ({detail})\n        ...{ctx}...")
    print(f"\nresults written to {OUT}")


if __name__ == "__main__":
    main()
