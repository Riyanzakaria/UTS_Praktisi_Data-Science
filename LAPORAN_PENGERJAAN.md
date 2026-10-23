# Laporan Pengerjaan per Role — Study Case: Article Named-Entity Recognition Service

> Dokumen ini menceritakan **alur pengerjaan langkah demi langkah** tiap role, dari awal
> sampai project selesai — cocok dipakai saat presentasi/defense ("ceritakan prosesnya").
> Untuk ringkasan akhir (artefak + angka metrik), lihat `PEMBAGIAN_PERAN.md`.

## Gambaran alur kerja keseluruhan

1. **Role 1 dulu:** profiling dataset → anotasi → split → training model.
2. **Setelah model jadi**, dua role jalan paralel: Role 2 (evaluasi) dan Role 3 (service + dokumentasi).
3. **Final:** verifikasi end-to-end + packaging.

---

## ROLE 1 — Data & NER Engineer

**Anggota: Riyan Zakaria Zulkarnain**

**Tujuan role:** menyiapkan data dan melatih model NER yang dipakai service.

Langkah yang dikerjakan:

1. **Profiling dataset.** Memuat 8 file CSV liputan media Maybank Malaysia
   (7 Sep – 7 Okt 2026) dengan `src/data_loader.py`. Temuan penting:
   - 1291 baris, tapi 445 duplikat `original_id` antar-file → **846 artikel unik**
     (duplikat dibuang, diambil kemunculan pertama).
   - 0 judul / body yang kosong.
   - Bahasa dominan Inggris (lalu Melayu, Mandarin, Indonesia).
   - Kolom `entities` **kosong 100%** → keputusan: anotasi dibuat manual dari nol,
     kolom itu tidak pernah dianggap ground truth.
   - Angka-angka ini disimpan di `data/profile_report.json` untuk README.

2. **Merancang skema label & pedoman anotasi.** Label: PERSON, ORG, GPE, DATE,
   EVENT, MONEY. Ditulis di `data/annotation_guidelines.md` beserta edge case:
   akronim ("FBM KLCI"), possessive, nama bertanda baca ("S&P 500", "Mr D.I.Y."),
   dan pemetaan label model baseline ke skema ini.

3. **Memilih 18 artikel** berbahasa Inggris yang beragam (sumber dan tanggal
   bervariasi, mencakup orang/organisasi/tempat/tanggal + nama-nama sulit).
   Aturan excerpt yang dipakai: `judul + "\n\n" + 300–600 karakter pertama body`.

4. **Anotasi span-level.** Pre-label otomatis pakai `en_core_web_sm`, lalu
   **review dan koreksi manual seluruh 179 span** — setiap offset dicek
   slice-back exact ke teks, tidak ada span yang overlap, semua label dalam skema.
   Hasil: `data/annotations.jsonl`.

5. **Split deterministik by article** (seed 42, teks satu artikel tidak boleh
   bocor ke split lain): **12 train / 3 val / 3 test** (span: 119 / 32 / 28).

6. **Training model** (`src/train_ner.py`). Fine-tuning komponen NER dari
   `en_core_web_sm`: 25 iterasi, dropout 0.2, batch compounding 4→32, seed 42
   (reproducible). F1 validasi dipantau tiap iterasi: naik 0.429 (iter 1) →
   **0.645 (iter 21, terbaik)** → 0.615 (iter 25, mulai overfit).
   Model terbaik disimpan ke `models/ner_model/` (±19 detik training).
   Riwayat tiap iterasi ada di `data/training_log.json`.

7. **Mencatat caveat jujur:** 2 span gold token-misaligned (`Malaysia` sebagai
   prefix `Malaysian`) diabaikan saat training (peringatan spaCy W030);
   evaluasi char-span tidak terpengaruh. Kelas PERSON paling sedikit di train (5 span).

---

## ROLE 2 — Evaluation Analyst

**Anggota: Hanif**

**Tujuan role:** membuktikan model hasil training benar-benar lebih baik dari
baseline, dengan metrik yang sahih dan analisis error yang konkret.

Langkah yang dikerjakan:

1. **Membangun evaluation set** (`src/build_evaluation_examples.py`):
   7 excerpt **baru** (tidak ada di 18 artikel anotasi) + 3 artikel test =
   **10 item held-out**, 103 gold span, tersimpan di `data/evaluation_examples.json`.
   Komposisi disengaja memenuhi syarat study case: campuran semua tipe entitas,
   entitas yang muncul berulang, nama ambigu/bertanda baca, **1 item tanpa
   entitas sama sekali**, dan contoh menantang (format tanggal Melayu
   seperti `1 Oktober`, `7.30 pagi`).

2. **Menulis `src/evaluate.py`** — evaluasi exact-span: prediksi benar hanya jika
   `(start, end, label)` sama persis dengan gold.

3. **Menjalankan evaluasi** baseline (`en_core_web_sm` mentah) vs model trained,
   di data val dan evaluation set. Hasil (tersimpan di `data/evaluation_results.json`):

   | Set | Baseline F1 | Trained F1 |
   |---|---|---|
   | Val (3 artikel) | 0.485 | **0.645** |
   | Eval-10 (10 item) | 0.298 | **0.492** |

   Per-label F1 (eval-10): ORG 0.198→**0.494**, GPE 0.300→**0.737**,
   PERSON 0.167→0.364, EVENT 0.000→0.200, DATE 0.681→0.632
   (baseline menang di DATE — model trained melewatkan format tanggal Melayu),
   MONEY 0.000→0.000 (kedua model tak memprediksi; hanya 3 span gold).

4. **Error analysis** — 4 contoh konkret beserta penyebabnya:
   - FP `Saingan`→ORG: kata benda umum Melayu yang dikapitalisasi ikut ke-tag.
   - FP `Durriyah Damian`→ORG (gold PERSON): konteks olahraga menarik label ORG.
   - FN `1 Oktober` / `7.30 pagi` (gold DATE): format tanggal/waktu Melayu tak dikenali
     (data training hampir semuanya tanggal Inggris).
   - FP `KUALA LUMPUR</strong`→GPE: sisa markup HTML bocor ke batas span.

5. **Pernyataan held-out** (ditulis eksplisit di README): 3 artikel test + 7 excerpt
   eval tidak pernah dipakai training, tidak pernah meng-update bobot, dan tidak
   memengaruhi pemilihan model — satu-satunya sinyal tuning adalah F1 val.

---

## ROLE 3 — Service & Docs Engineer

**Anggota: Arrizal**

**Tujuan role:** membungkus model jadi layanan web yang bisa didemo, menguji
semuanya otomatis, dan mendokumentasikan seluruh project.

Langkah yang dikerjakan:

1. **Membangun service Flask** (`src/app.py`):
   - `POST /extract-entities` menerima JSON `{"title","body"}`; teks yang dianalisis
     = `title + "\n\n" + body`. Validasi → **HTTP 400** (body JSON) untuk: judul+body
     kosong, field bukan string, JSON rusak, gabungan > 100.000 karakter.
     Sukses → **HTTP 200** `{"entities":[{"text","label","start","end"}],
     "meta":{"entity_count":N,"model":"maybank-ner-1.0"}}` — deterministik untuk
     input dan versi model yang sama.
   - `GET /` panel browser: field judul + body, tombol submit, **loading state**,
     tabel hasil (teks, label, start, end), error banner yang tidak bikin halaman
     crash, dan highlight span entitas langsung di teks (`<mark>` per label).

2. **Verifikasi service terhadap model asli** (`models/ner_model`), bukan stub.

3. **Menulis 12 automated tests** (`tests/test_api.py`, pytest) — mencakup: input
   kosong → 400, JSON rusak → 400, field non-string → 400, input > 100rb char → 400,
   request valid → 200 dengan offset yang slice-back exact ke teks, entitas yang
   diharapkan pada contoh tetap, determinisme (panggilan ulang = output identik),
   dan smoke test panel. **12/12 passed**, diverifikasi independen.

4. **Menulis `README.md` lengkap**: instalasi, perintah startup, contoh request &
   response API penuh, diagram arsitektur, alasan pemilihan model + pemetaan label
   baseline, ringkasan anotasi/split/training, tabel hasil evaluasi, **jawaban 8
   pertanyaan study case**, limitasi, dan pembahasan responsible data use.
   Plus `requirements.txt` yang ter-pin versinya.

5. **Verifikasi end-to-end**: server dinyalakan, excerpt asli dari test set
   di-POST → 200 dengan 9 entitas dan offset terverifikasi exact; input kosong →
   400; panel di-GET → ter-render lengkap (form, tabel, highlight, error banner).

6. **Packaging**: tidak ada salinan CSV dataset di `project/` (sesuai larangan
   study case), tidak ada hard-coded entity untuk contoh demo, tidak ada secret;
   arsip `project-ner-maybank.zip` dibuat untuk pengumpulan.

---

## Ketergantungan antar role (penting untuk presentasi)

- **Role 1 → Role 2 & Role 3.** Role 2 butuh model final untuk dievaluasi;
  Role 3 butuh model final untuk diserve. Keduanya baru bisa full-speed setelah
  Role 1 selesai training.
- **Role 2 dan Role 3 paralel** setelah model tersedia.
- Keputusan desain terbesar ada di Role 1 (kolom entities kosong → anotasi manual;
  pilih fine-tuning spaCy karena dataset kecil); temuan paling jujur ada di
  Role 2 (model kalah di tanggal Melayu, MONEY nol besar); bukti "bisa dipakai"
  ada di Role 3 (service live + 12 test hijau).
