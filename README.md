# Maybank News NER — Entity Extraction Service

Named-entity recognition for Maybank news monitoring: a spaCy pipeline fine-tuned on
18 human-annotated English news articles identifies people, organizations, places,
dates, events, and monetary amounts in a headline plus body, and serves the results
through a small Flask web service (`POST /extract-entities` API plus an interactive
browser panel). The trained model reaches exact-span F1 0.645 on a validation set
and 0.492 on a held-out evaluation set, clearly outperforming the unmapped baseline
`en_core_web_sm` (F1 0.298 on held-out data).

## Installation

```bash
python -m venv venv              # or reuse the existing ../venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Startup

```bash
cd project
../venv/bin/python src/app.py     # Flask serves on http://127.0.0.1:5000
```

Other entry points:

```bash
../venv/bin/python src/train_ner.py     # train the model -> models/ner_model
../venv/bin/python src/evaluate.py      # evaluate baseline vs trained -> data/evaluation_results.json
../venv/bin/python -m pytest tests/     # API test suite (uses the real model)
```

## API usage

### `POST /extract-entities`

Request body (JSON, both fields required strings, at least one non-empty; combined
analyzed text capped at 100,000 characters):

```json
{
  "title": "Maybank 2Q profit rises",
  "body": "Maybank posted a net profit of RM2.69 billion in Kuala Lumpur on Wednesday."
}
```

Example response (`200 OK`; every `text[start:end]` slices back to the analyzed
text `title.strip() + "\n\n" + body.strip()`):

```json
{
  "entities": [
    {"end": 7,  "label": "ORG",   "start": 0,  "text": "Maybank"},
    {"end": 32, "label": "ORG",   "start": 25, "text": "Maybank"},
    {"end": 70, "label": "MONEY", "start": 56, "text": "RM2.69 billion"},
    {"end": 86, "label": "GPE",   "start": 74, "text": "Kuala Lumpur"},
    {"end": 99, "label": "DATE",  "start": 90, "text": "Wednesday"}
  ],
  "meta": {"entity_count": 5, "model": "maybank-ner-1.0"}
}
```

Error responses (`400`): both-empty input, malformed JSON, non-string fields, or
input exceeding the character limit — all return `{"error": "..."}`.

### `GET /`

Interactive browser panel: paste a title and body, submit, and inspect the
entity table plus highlighted annotated text. The panel rebuilds the same
`title + "\n\n" + body` string client-side, so highlights align with the API offsets.

## Architecture

```
                     ┌────────────────────────────┐
  browser panel ──▶  │  Flask app (src/app.py)    │  GET  /
  API client    ──▶  │  POST /extract-entities    │
                     └────────────┬───────────────┘
                                  │ title.strip() + "\n\n" + body.strip()
                                  ▼
                     ┌────────────────────────────┐
                     │ spaCy pipeline             │  maybank-ner-1.0
                     │ (models/ner_model)         │  tok2vec → tagger → parser
                     │ NER component              │  → senter → ner
                     └────────────┬───────────────┘
                                  │ doc.ents -> sorted, deduped
                                  ▼
                     ┌────────────────────────────┐
                     │ JSON: [{text, label,       │
                     │         start, end}] + meta│
                     └────────────────────────────┘
```

Data flow: the panel or API client sends `{title, body}`; the service concatenates
them into one analyzed string, runs the NER pipeline once, sorts entities by
`(start, end)`, removes exact duplicates, and returns deterministic JSON.
The model is loaded exactly once at startup (`NER_MODEL_DIR` env var can point at
any spaCy pipeline directory, e.g. for smoke tests).

## Model choice — and why

The model is spaCy's `en_core_web_sm` **fine-tuned on our own annotations**
(`models/ner_model`, identifier `maybank-ner-1.0`).

- **Transfer learning fits 12 training documents.** Training a token classifier from
  scratch on 119 spans would underfit badly; a pre-trained English NER already knows
  generic PERSON/ORG/GPE/DATE patterns, and fine-tuning adapts it to finance-domain
  conventions (e.g. `FTSE Bursa Malaysia KLCI (FBM KLCI)` as two ORG spans,
  `S&P 500` as one ORG span).
- **CPU-friendly and fast.** Training took ~19 s; inference is milliseconds per
  request — no GPU needed, which fits a classroom deployment.
- **Built-in character offsets.** spaCy returns `ent.start_char`/`ent.end_char`
  directly, which is exactly what the API contract needs for the annotated-text
  highlight panel — no offset reconstruction hacks.
- **Deterministic inference.** The pipeline has no randomness at predict time, so
  identical requests return byte-identical JSON (covered by `tests/test_api.py`).

### Baseline mapping (raw `en_core_web_sm` → our label scheme, used for comparison)

| spaCy label | Mapped to | Notes |
|---|---|---|
| PERSON | PERSON | — |
| ORG | ORG | — |
| GPE, LOC | GPE | both are places |
| DATE, TIME | DATE | both are temporal expressions |
| EVENT | EVENT | — |
| MONEY | MONEY | — |
| NORP, FAC, PRODUCT, WORK_OF_ART, LAW, LANGUAGE, QUANTITY, ORDINAL, CARDINAL, PERCENT | dropped | outside the scheme |

## Annotation summary

- **18 English news articles**, annotated at `title + "\n\n" + excerpt` granularity
  (excerpt = first ~300–600 characters of body, cut at a sentence boundary).
- **Human-reviewed**: the baseline model produced candidate spans; a human reviewer
  checked every span — accept, fix boundaries, fix label, delete, or add missing
  spans. Model suggestions were never accepted blindly.
- **179 spans total** across the scheme `PERSON, ORG, GPE, DATE, EVENT, MONEY`.
- Full guidelines (definitions, edge cases, possessives, acronym handling, repeated
  mentions, no-entity texts): [`data/annotation_guidelines.md`](data/annotation_guidelines.md).

## Splits

Split **by article** with deterministic seed 42 (no article appears in two splits):

| Split | Articles | Spans |
|---|---|---|
| train | 12 | 119 |
| val | 3 | 32 |
| test | 3 | 28 |

Train label counts: ORG 43, DATE 30, GPE 16, EVENT 14, MONEY 11, PERSON 5.

## Training settings and duration

- 25 iterations, dropout 0.2, compounding batch sizes 4 → 32, seed 42.
- Model selection: after each iteration, exact-span F1 on the 3 val records;
  the iteration with the highest val F1 was kept — **iteration 21, val F1 0.645**
  (ties resolved to the earliest iteration).
- Duration: **~19 s training** (≈26 s wall clock), CPU-only.

## Evaluation

Metric for everything below: **exact (start, end, label) match** — a prediction
counts as correct only if its character offsets *and* label equal the gold span.

### Validation set (3 items, 32 gold spans)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (`en_core_web_sm` + mapping) | 0.471 | 0.500 | 0.485 |
| Trained (`maybank-ner-1.0`) | 0.667 | 0.625 | 0.645 |

### Held-out evaluation set (10 items, 103 gold spans)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (`en_core_web_sm` + mapping) | 0.295 | 0.301 | 0.298 |
| Trained (`maybank-ner-1.0`) | 0.534 | 0.456 | 0.492 |

### Per-label F1 on the held-out set (baseline → trained)

| Label | Baseline | Trained |
|---|---|---|
| PERSON | 0.167 | 0.364 |
| ORG | 0.198 | 0.494 |
| GPE | 0.300 | 0.737 |
| DATE | 0.681 | 0.632 |
| EVENT | 0.000 | 0.200 |
| MONEY | 0.000 | 0.000 |

Fine-tuning improves five of six labels — GPE nearly triples (0.300 → 0.737) and
ORG more than doubles (0.198 → 0.494). DATE drops slightly (0.681 → 0.632):
the baseline already handled canonical English dates well, and the fine-tuned
model trades some of that generality for domain patterns it saw in training.
MONEY stays at zero on held-out data — a data-sparsity problem (11 training spans),
not a modeling problem.

### Error analysis (concrete examples)

1. **FP: `'Saingan'` → ORG** in the Malay passage "…7.30 pagi tadi. *Saingan* kali ini
   turut menampilkan…" — a Malay common noun ("competition") over-tagged as an
   organization. Fine-tuning on English financial text makes capitalized-looking
   Malay nouns trigger the ORG classifier.
2. **FP: `'Durriyah Damian'` → ORG (gold PERSON)** in "…Allycia Gan, Liyana Durisic,
   *Durriyah Damian* dan Kelly Tan…" — a person name misclassified as ORG. The
   training set has only 5 PERSON spans, so the model defaults unfamiliar name
   shapes to the majority class ORG.
3. **FN: `'1 Oktober'` / `'7.30 pagi'`** (gold DATE, predicted nothing) — Malay
   date forms are missed entirely. The model was tuned on English text and never
   saw Malay month names or `HH.MM pagi/petang` time patterns.
4. **FP: `'KUALA LUMPUR</strong'` → GPE** — HTML markup leaked into the span
   boundary. Source texts contain residual HTML (`<strong>` tags); the annotator's
   gold boundary was clean but the model extended the GPE span over the markup.
   Fixing this belongs upstream (strip HTML before annotation and inference).

### Held-out statement

The 3 test articles in `data/annotations.jsonl` and the 7 eval excerpts in
`data/evaluation_examples.json` were **never used in training or tuning**; they
did not influence model weights or iteration selection. Only val F1 on the 3 val
records guided iteration selection.

## Case questions

### 1. What makes a span a named entity here?

A span is an entity if it satisfies *both* conditions: (a) it is a **proper name
or concrete referring expression** belonging to one of the six supported types
(PERSON, ORG, GPE, DATE, EVENT, MONEY) as defined in the annotation guidelines —
e.g. `Maybank` (ORG), `Kuala Lumpur` (GPE), `RM2.69 billion` (MONEY); and (b) its
**boundaries are exact** — including legal suffixes (`Sdn Bhd`, `Berhad`) and
excluding titles (`Mr`) and possessive `'s`. Generic nouns (`bank`, `market`),
bare statistics (`306 counters`), percentages, and durations are *not* entities.

### 2. Why combine title + body, and what is lost otherwise?

The analyzed text is `title.strip() + "\n\n" + body.strip()` because headlines
carry dense entity information (`Maybank 2Q profit rises` alone yields ORG +
MONEY cues) and the model needs that context. Running the body alone would lose:
(a) headline-only entities, (b) title→body coreference context the model uses to
disambiguate (a name in the title primes the NER for the same name in the body),
and (c) a single stable offset space — the panel's highlight rendering depends on
one string. The only cost is a minor domain shift: headlines are compressed and
fragmentary, so the model must be robust to both terse and full-sentence styles.

### 3. Supported labels and user mapping

The service supports exactly six labels: PERSON, ORG, GPE, DATE, EVENT, MONEY.
A "user mapping" (the baseline comparison in the README section above) translates
raw `en_core_web_sm` labels into this scheme — GPE/LOC → GPE, DATE/TIME → DATE,
everything else not in the scheme (NORP, FAC, PERCENT, …) is dropped. Users of
the trained model need no mapping: its output labels *are* the scheme.

### 4. Tokenization, punctuation, capitalization, language effects

- **Tokenization/punctuation**: spaCy's tokenizer splits on whitespace and
  punctuation, so hyphenated or punctuated names (`S&P 500`, `Mr D.I.Y.`, `S&P`)
  are handled as multi-token spans; the annotation rules explicitly keep `&` and
  internal periods inside one span.
- **Capitalization**: the model leans on capitalization as a signal, which causes
  false positives on capitalized common nouns (error example 1: `Saingan`) and
  correctly groups case variants of the same name (`Bursa Malaysia` vs
  `BURSA MALAYSIA` → both ORG).
- **Language**: the pipeline is English-tuned (`en_core_web_sm`). It degrades on
  Malay text — Malay date forms are missed (error example 3) and Malay common
  nouns are over-tagged (error example 1). This is a known, documented limitation.

### 5. Text vs label vs offsets

- **Text** is the raw substring that was found — human-readable but ambiguous
  (`Maybank` occurs twice in the example response).
- **Label** is the semantic type (ORG, GPE, …) — the *meaning* of the span.
- **Offsets** (`start`, `end`) are zero-based, end-exclusive character positions
  into the exact analyzed string. They are the ground truth for evaluation
  (exact-match metric requires identical offsets *and* label) and for the panel
  (they position the highlights). Every returned offset is verifiable:
  `text[start:end] == entity["text"]` (asserted in the test suite).

### 6. Repeated entities with spelling/capitalization variants

The annotation guidelines require **every occurrence** to be annotated, even in
one text — the API likewise returns every mention (the example response lists
`Maybank` at 0–7 and 25–32). Downstream consumers must decide the grouping
policy: exact-string matching is the safe default; fuzzy normalization
(`Bursa Malaysia` ≈ `BURSA MALAYSIA`, acronyms like `FBM KLCI`) needs a gazetteer
or coreference layer the service deliberately does not include.

### 7. Which errors matter most for news monitoring, and improvements

For news monitoring, **missed entities (FNs) hurt more than false positives**:
an untagged `Maybank` mention is an untracked story, while a stray FP is visible
and dismissible. The costliest concrete gaps: (a) missed Malay dates — fix by
adding Malay-language training examples or a Malay date regex post-processor;
(b) PERSON↔ORG confusion from 5 training PERSON spans — fix with targeted
annotation of person names in financial context; (c) HTML leaking into spans —
fix upstream by stripping markup in the data loader before annotation *and*
inference; (d) MONEY recall at zero — annotate more monetary amounts, especially
in Malay/Indonesian formats (`RM`, `sen`, `US$`).

### 8. Privacy, copyright, bias, responsible use

See [Responsible data use](#responsible-data-use) below. Key points: the dataset
contains no private personal data (public news about a public company); source
articles are third-party copyrighted text, so the dataset must stay inside the
coursework; model bias reflects the training data (English finance news — weak
on Malay/Indonesian, on women/non-Western names, and on small labels like
MONEY); and the model is a **prototype**, not production software — its output
should be human-verified before any real monitoring decision.

## Limitations

- **Small training set**: 12 documents / 119 spans — enough for fine-tuning, far
  from a production corpus; rare labels (PERSON 5, MONEY 11) starve.
- **English-only tuning**: Malay date forms and Malay common nouns are handled
  poorly (see error analysis).
- **MONEY/EVENT starved**: zero held-out F1 for MONEY; EVENT barely learned.
- **HTML noise in source texts** leaks into predicted span boundaries.
- **2 token-misaligned gold spans were ignored in training** (spaCy W030
  warnings) — they did not contribute to learning.
- **Prototype, not production**: built for coursework; no rate limiting, auth,
  logging, or monitoring — do not deploy as-is behind the internet.

## Responsible data use

- **No private data.** All 18 articles are publicly published English news about
  a public company (Maybank). No private communications, user data, or sensitive
  personal information is included.
- **Copyright.** The source articles are third-party copyrighted material used
  for a non-commercial coursework exercise. The dataset (including any CSV
  exports) must not be published, redistributed, or used commercially; no dataset
  CSV is committed to this repository (see Project tree).
- **Bias.** The model's behavior reflects its narrow training distribution:
  English financial news, heavily skewed toward ORG and DATE mentions. It
  systematically underperforms on Malay/Indonesian text, unfamiliar (non-Western)
  name shapes, and rare labels — users must not mistake its confidence for
  correctness on out-of-distribution text.
- **No automated decisions.** The service is a demo of NER fine-tuning and
  evaluation; any real news-monitoring use requires human review of extracted
  entities before acting on them.

## Tests

```bash
../venv/bin/python -m pytest tests/
```

`tests/test_api.py` exercises the real model through Flask's test client:
empty/whitespace input → 400, malformed/non-object JSON → 400, non-string or
missing fields → 400, over-limit input → 400, valid request → 200 with offsets
verified to slice back exactly (including a fixed GPE `Kuala Lumpur` check),
sorted/deduplicated entities, byte-identical determinism, and the browser panel
markup. 12 tests, all passing.

## Project tree

```
project/
├── README.md
├── requirements.txt
├── src/
│   ├── app.py                 # Flask service (POST /extract-entities, GET / panel)
│   ├── train_ner.py           # fine-tunes en_core_web_sm -> models/ner_model
│   ├── evaluate.py            # baseline vs trained evaluation
│   ├── data_loader.py         # annotation JSONL -> training records
│   └── build_evaluation_examples.py
├── data/
│   ├── annotations.jsonl            # 18 human-reviewed articles (train/val/test)
│   ├── annotation_guidelines.md     # label definitions and edge cases
│   ├── evaluation_examples.json     # 10 held-out eval items
│   ├── evaluation_results.json      # baseline vs trained metrics
│   ├── training_log.json            # per-iteration val F1, best = iteration 21
│   └── profile_report.json          # source corpus profiling
├── models/
│   └── ner_model/             # trained spaCy pipeline (maybank-ner-1.0)
└── tests/
    └── test_api.py            # 12 API tests against the real model
```

Note: no dataset CSV is committed anywhere under `project/` — the raw article
CSVs live outside the repository.
