# Annotation Guidelines — Article NER Service Study Case

Label scheme: `PERSON`, `ORG`, `GPE`, `DATE`, `EVENT`, `MONEY`.
Text unit: `title + "\n\n" + excerpt`, where excerpt = first ~300–600 chars of body,
cut at a sentence boundary when possible (exact rule in `src/data_loader.py:make_text`).
Offsets are zero-based character offsets into this combined text, end-exclusive.

## Label definitions

- **PERSON** — a named individual: full name or recognizable partial name.
  Ex: `Thong Pak Leng`, `Anwar Ibrahim`. Titles ("Mr", "Dr", "Datuk") are NOT included
  in the span unless they are part of the common written form.
- **ORG** — companies, banks, agencies, institutions, exchanges and market indices
  treated as organizations. Include legal suffixes: `Rakuten Trade Sdn Bhd`,
  `Press Metal Aluminium Berhad`, `Hong Leong Bank`, `CIMB`, `Bursa Malaysia`,
  `FTSE Bursa Malaysia KLCI`, `S&P 500` (one span, keep the `&`), `FBM KLCI`.
- **GPE** — geopolitical entities: countries, cities, states, regions.
  Ex: `Malaysia`, `Kuala Lumpur`, `Hong Kong`, `US`. `Wall Street` is GPE
  (metonymic place). `Brent` (oil benchmark) is NOT an entity.
- **DATE** — date/time expressions, absolute or relative.
  Ex: `Wednesday`, `Oct 7`, `October 7, 2026`, `9:11 AM`, `Tuesday` (in `Tuesday's`,
  strip the possessive `'s`). Bare years are annotated only when they function as
  a date reference.
- **EVENT** — named events: conferences, summits, named programmes, tournaments.
  Ex: `PNM CUP 2026`. A generic "the launch" is NOT an event.
- **MONEY** — monetary amounts. Ex: `RM21.98`, `RM150.56 million`, `10 sen`,
  `US$100`, `RM1.24`. Percentages are NOT annotated (out of scheme).

## What is NOT an entity

- Generic lowercase nouns: `bank`, `index`, `market`, `counters`.
- Bare statistics: `221 to 180`, `306 counters`, `2,083 untraded`, `3.68 points`.
- Durations / yields: `10-year yield`, `5.286%` (percent is out of scheme).
- Commodity names used generically: `Brent` when referring to the oil price.

## Edge cases / disagreements (baseline model vs. final guidelines)

1. **Acronym after full name** — `FTSE Bursa Malaysia KLCI (FBM KLCI)`: annotate the
   full name AND the acronym as two separate ORG spans.
2. **Possessives** — `Tuesday's close` → annotate `Tuesday` only (strip `'s`).
3. **Punctuated names** — `Mr D.I.Y.` is one ORG span including the periods;
   `S&P 500` is one ORG span including the ampersand.
4. **Repeated mentions** — every occurrence is annotated, even within one text.
   This is required by the case (repeated-entity handling).
5. **Capitalization variants** — `Bursa Malaysia` vs `BURSA MALAYSIA` → same label ORG.
6. **Org-vs-product ambiguity** — product/brand mentions in market context
   (`iPhone` would be a product; in this corpus index names like `FBM KLCI`
   are ORG). When genuinely ambiguous, prefer ORG for listed market instruments
   and record the disagreement in the annotation file's `notes`.
7. **No-entity texts** — a text with no named entity gets `"entities": []`.
   This is a valid, required case.
8. **Baseline-model corrections applied during review**: en_core_web_sm tends to
   (a) split `Rakuten Trade Sdn Bhd` into two ORGs, (b) label only `Bursa` or miss
   `FTSE Bursa Malaysia KLCI` entirely, (c) split `S&P 500`, (d) miss `Mr D.I.Y.`.
   All spans are corrected by the human reviewer; the model's suggestion is never
   accepted blindly.

## Baseline label mapping (en_core_web_sm → our scheme, for comparison)

| spaCy label | Mapped to | | spaCy label | Mapped to |
|---|---|---|---|---|
| PERSON | PERSON | | DATE, TIME | DATE |
| ORG | ORG | | EVENT | EVENT |
| GPE, LOC | GPE | | MONEY | MONEY |
| NORP, FAC, PRODUCT, WORK_OF_ART, LAW, LANGUAGE, QUANTITY, ORDINAL, CARDINAL, PERCENT | dropped (out of scheme) | | | |

## Review process

Annotator runs `en_core_web_sm` to get candidate spans, then checks EVERY span:
accept, fix boundaries, fix label, delete, or add missing spans. The final JSONL
is the annotator's own judgement; model suggestions are a starting point only.
