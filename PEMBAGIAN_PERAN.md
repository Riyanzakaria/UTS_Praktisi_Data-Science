# Pembagian Peran — Study Case: Article Named-Entity Recognition Service

Kelompok terdiri dari **3 anggota**. Setiap role di bawah dikerjakan oleh satu anggota
dan seluruh artefaknya tercatat dengan path file, sehingga tiap anggota dapat
mempresentasikan bagiannya secara mandiri.

Ringkasan proyek: layanan web (Flask) yang menerima judul + isi artikel berita dan
mengembalikan named entity (teks, label, offset karakter) lewat `POST /extract-entities`.
Model NER adalah hasil **fine-tuning** `en_core_web_sm` pada 12 artikel beranotasi
dari dataset liputan media Maybank Malaysia (7 Sep – 7 Okt 2026), bukan model
pretrained mentah dan bukan sekadar rules.

---

## ROLE 1 — Data & NER Engineer

**Anggota: Riyan Zakaria Zulkarnain**

**Ruang lingkup:** memuat dan memprofilkan dataset, merancang skema label dan
pedoman anotasi, menganotasi 18 artikel (dengan review manual penuh atas
pre-label model), membagi train/val/test secara deterministik, melatih /
fine-tuning model NER spaCy, dan menyimpan artefak model final.

**Artefak:**

| File | Isi |
|---|---|
| `project/src/data_loader.py` | `clean()`, `make_text()` (aturan excerpt judul + `\n\n` + 300–600 karakter body), `load_articles()` (dedupe by `original_id`), `profile()` |
| `project/data/profile_report.json` | Angka profil dataset |
| `project/requirements.txt` | `spacy==3.8.16`, `flask==3.1.3`, `pandas==3.0.6`, `pytest==9.1.1` (+ `en_core_web_sm==3.8.0` via `spacy download`) |
| `project/data/annotation_guidelines.md` | Skema label PERSON/ORG/GPE/DATE/EVENT/MONEY, definisi, edge case (akronim, possessive, nama bertanda baca), pemetaan label baseline |
| `project/data/annotations.jsonl` | 18 artikel beranotasi span-level, format `{article_id, source_file, row_index, split, text, entities:[{start,end,label}], notes}` |
| `project/src/train_ner.py` | Script training reproduksibel (seed 42, 25 iterasi, dropout 0.2, batch compounding 4→32) |
| `project/data/training_log.json` | Riwayat F1 validasi per iterasi + iterasi terbaik |
| `project/models/ner_model/` | Pipeline spaCy final (iterasi terbaik) — dipakai service sebagai `maybank-ner-1.0` |

**Metrik & fakta kunci (untuk presentasi):**
- Dataset: 1291 baris → **846 unik** (445 duplikat `original_id`, diambil kemunculan pertama); 0 judul/body kosong.
- Bahasa (unik): en 656, ms 146, zh 28, id 14, vi 1, es 1 → dikerjakan subset **Inggris** (terdokumentasi).
- Kolom `entities` **kosong 100%** → anotasi dibuat dari nol, tidak pernah dianggap ground truth.
- Split by article, seed 42: **12 train / 3 val / 3 test**; span: train 119, val 32, test 28 (total 179). Imbalance train: ORG 43, DATE 30, GPE 16, EVENT 14, MONEY 11, PERSON 5.
- Training: durasi **±19 dtk** (±26 dtk wall); F1 val naik 0.429 (iter 1) → **0.645 (iter 21, terbaik)** → 0.615 (iter 25, overfit ringan).
- Caveat jujur: 2 span gold token-misaligned (`Malaysia` sebagai prefix `Malaysian`) diabaikan saat training (peringatan spaCy W030); evaluasi char-span tidak terpengaruh.

---

## ROLE 2 — Evaluation Analyst

**Anggota: Hanif**

**Ruang lingkup:** membangun evaluation set 10 item yang held-out, menghitung
metrik exact-span precision/recall/F1 (overall + per-label) untuk baseline vs
model trained pada data val dan evaluation set, serta menulis error analysis
dengan contoh FP/FN konkret. Menjamin dan menyatakan bahwa data test/eval tidak
pernah memengaruhi training.

**Artefak:**

| File | Isi |
|---|---|
| `project/src/build_evaluation_examples.py` | Builder evaluation set (offset terverifikasi mesin) |
| `project/data/evaluation_examples.json` | 10 item held-out (3 artikel test + 7 excerpt baru), 103 gold span |
| `project/src/evaluate.py` | Evaluasi exact-span `(start, end, label)` baseline vs trained |
| `project/data/evaluation_results.json` | Seluruh angka metrik (val + eval10, overall + per-label) |

**Metrik kunci (untuk presentasi):**

| Set | Model | Precision | Recall | F1 |
|---|---|---|---|---|
| Val (3 artikel, 32 span) | baseline `en_core_web_sm` | 0.471 | 0.500 | 0.485 |
| Val | **trained** | 0.667 | 0.625 | **0.645** |
| Eval-10 (10 item, 103 span) | baseline | 0.295 | 0.301 | 0.298 |
| Eval-10 | **trained** | 0.534 | 0.456 | **0.492** |

- Per-label F1 (eval-10, baseline → trained): PERSON 0.167→0.364, **ORG 0.198→0.494**,
  **GPE 0.300→0.737**, DATE 0.681→0.632 (baseline menang — model trained melewatkan
  format tanggal Melayu seperti `1 Oktober`, `7.30 pagi`), EVENT 0.000→0.200,
  MONEY 0.000→0.000 (tidak ada model yang memprediksi; hanya 3 span gold).
- Error analysis (contoh konkret): FP `Saingan`→ORG (kata benda Melayu dikapitalisasi
  over-tag); FP `Durriyah Damian`→ORG (gold PERSON, konteks olahraga menarik label);
  FN `1 Oktober`/`7.30 pagi` (gold DATE, format Melayu tak dikenali);
  FP `KUALA LUMPUR</strong`→GPE (markup HTML bocor ke batas span).
- **Pernyataan held-out:** 3 artikel test + 7 excerpt eval tidak pernah dipakai
  training, tidak pernah update bobot, dan tidak memengaruhi pemilihan model —
  satu-satunya sinyal tuning adalah F1 val pada 3 artikel val.

---

## ROLE 3 — Service & Docs Engineer

**Anggota: Arrizal**

**Ruang lingkup:** verifikasi service Flask + panel browser, menulis automated
tests (pytest, semuanya hijau), menulis README lengkap (instalasi, API,
arsitektur, alasan pemilihan model, metrik, jawaban 8 pertanyaan study case,
limitasi, responsible use), verifikasi end-to-end (server + curl + panel),
dan kerapian packaging.

**Artefak:**

| File | Isi |
|---|---|
| `project/src/app.py` | Flask: `POST /extract-entities` (validasi → 400 JSON; sukses → 200 entities + meta), `GET /` panel (form, loading state, tabel hasil, error banner, highlight `<mark>` per label) |
| `project/tests/test_api.py` | **12 test pytest, 12 passed** — 400 (kosong, JSON rusak, non-string, >100.000 char), 200 (offset slice-back exact, GPE `Kuala Lumpur`, determinisme byte-identical), smoke test panel |
| `project/README.md` | Dokumen lengkap: instalasi, startup, contoh request/response penuh, diagram arsitektur ASCII, alasan pemilihan model, ringkasan anotasi/split/training, tabel evaluasi, **jawaban 8 pertanyaan study case**, limitasi, responsible data use |

**Metrik & fakta kunci (untuk presentasi):**
- Kontrak API: input `{"title","body"}` (string); teks analisis = `title.strip() + "\n\n" + body.strip()`; batas 100.000 karakter (`MAX_INPUT_CHARS`); output `{"entities":[{"text","label","start","end"}], "meta":{"entity_count":N,"model":"maybank-ner-1.0"}}`; deterministik untuk input + versi model yang sama.
- Verifikasi end-to-end: excerpt test-set asli → 200 dengan 9 entity & offset terverifikasi slice-back exact; input kosong → 400; `GET /` me-render panel (form, error banner, tabel, highlight).
- Packaging: tidak ada salinan CSV dataset di `project/` (sesuai larangan study case); tidak ada hard-coded entity; tidak ada secret.

---

## Cara menjalankan (untuk demo)

```bash
cd project
../venv/bin/python -m pip install -r requirements.txt
../venv/bin/python -m spacy download en_core_web_sm
../venv/bin/python src/app.py        # panel: http://localhost:5000
../venv/bin/python -m pytest tests/  # 12 passed
```
