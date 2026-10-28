"""Flask service: NER extraction panel for the Maybank NER coursework project.

Loads a trained spaCy pipeline once at startup and exposes:
  - POST /extract-entities  JSON API
  - GET  /                 interactive browser panel

Model directory is ``models/ner_model`` relative to the project root
(the directory containing this file's ``src/`` folder), overridable with
the ``NER_MODEL_DIR`` environment variable (used by smoke tests, e.g.
pointing at ``en_core_web_sm``).
"""

import html
import json
import os
import sys

from flask import Flask, jsonify, request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Maximum number of characters of the combined analyzed text that the
#: service will process in one request.
MAX_INPUT_CHARS = 100000

#: Model identifier reported in API responses. This is the name/version of
#: the trained pipeline produced by the later training step.
MODEL_NAME = "maybank-ner-1.0"


def _project_root() -> str:
    """Return the project root (parent of this file's ``src/`` directory)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_model_dir() -> str:
    """Resolve the default model directory (``models/ner_model`` under root)."""
    return os.path.join(_project_root(), "models", "ner_model")


def _load_model():
    """Load the spaCy pipeline once. Exit with a clear message if missing."""
    import spacy  # imported late so startup errors read cleanly

    model_dir = os.environ.get("NER_MODEL_DIR") or _default_model_dir()
    if not os.path.isdir(model_dir):
        sys.stderr.write(
            f"ERROR: NER model directory not found: {model_dir}\n"
            "The model is produced by a later training pipeline step. "
            "Train the model first, or point NER_MODEL_DIR at a valid spaCy "
            "pipeline directory.\n"
        )
        sys.exit(1)
    return spacy.load(model_dir)


# Load the trained pipeline exactly once at startup; exposed module-level
# for tests and reuse.
nlp = _load_model()

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Entity extraction logic
# ---------------------------------------------------------------------------

def build_text(title: str, body: str) -> str:
    """Build the analyzed text from title and body.

    The offsets returned by the API are character offsets into this exact
    string: ``title.strip() + "\\n\\n" + body.strip()``.
    """
    return title.strip() + "\n\n" + body.strip()


def extract_entities(title: str, body: str) -> list:
    """Extract entities from ``title`` and ``body``.

    Returns a list of dicts ``{"text", "label", "start", "end"}`` where
    ``start``/``end`` are char offsets into ``build_text(title, body)``.
    Entities are sorted by ``(start, end)`` and exact-duplicate spans are
    deduplicated, so the result is deterministic for a given input.
    """
    text = build_text(title, body)
    doc = nlp(text)

    seen = set()
    entities = []
    for ent in doc.ents:
        key = (ent.start_char, ent.end_char, ent.label_, ent.text)
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
        )
    entities.sort(key=lambda e: (e["start"], e["end"]))
    return entities


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Serve the interactive extraction panel (inline HTML)."""
    return INDEX_HTML


@app.route("/extract-entities", methods=["POST"])
def extract_entities_route():
    """JSON endpoint: {title, body} -> {entities, meta}."""
    data = request.get_json(force=False, silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({"error": "invalid JSON body"}), 400

    title = data.get("title")
    body = data.get("body")

    if not isinstance(title, str) or not isinstance(body, str):
        return jsonify({"error": "title and body must both be present and be strings"}), 400

    if not title.strip() and not body.strip():
        return jsonify({"error": "title and body are both empty"}), 400

    text = build_text(title, body)
    if len(text) > MAX_INPUT_CHARS:
        return (
            jsonify(
                {
                    "error": (
                        f"input too long: {len(text)} characters, "
                        f"maximum is {MAX_INPUT_CHARS}"
                    )
                }
            ),
            400,
        )

    entities = extract_entities(title, body)
    return (
        jsonify(
            {
                "entities": entities,
                "meta": {"entity_count": len(entities), "model": MODEL_NAME},
            }
        ),
        200,
    )


# ---------------------------------------------------------------------------
# Browser panel (inline template)
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Maybank NER — Named Entity Extraction Service</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-page: #f8fafc;
    --card-bg: #ffffff;
    --text-main: #0f172a;
    --text-muted: #64748b;
    --border-color: #e2e8f0;
    --primary: #0284c7;
    --primary-hover: #0369a1;
    --primary-light: #e0f2fe;
    --accent: #f59e0b;
    
    /* Entity Colors */
    --c-org-bg: #dbeafe; --c-org-txt: #1e40af; --c-org-border: #93c5fd;
    --c-person-bg: #fce7f3; --c-person-txt: #9d174d; --c-person-border: #f9a8d4;
    --c-gpe-bg: #dcfce7; --c-gpe-txt: #166534; --c-gpe-border: #86efac;
    --c-date-bg: #fef3c7; --c-date-txt: #92400e; --c-date-border: #fcd34d;
    --c-event-bg: #f3e8ff; --c-event-txt: #6b21a8; --c-event-border: #d8b4fe;
    --c-money-bg: #ffedd5; --c-money-txt: #9a3412; --c-money-border: #fdba74;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    background-color: var(--bg-page);
    color: var(--text-main);
    line-height: 1.6;
    padding: 2rem 1rem 4rem;
  }

  .container {
    max-width: 1040px;
    margin: 0 auto;
  }

  /* Header */
  .header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #ffffff;
    padding: 2.25rem 2rem;
    border-radius: 16px;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .header::after {
    content: '';
    position: absolute;
    top: -50px;
    right: -50px;
    width: 220px;
    height: 220px;
    background: radial-gradient(circle, rgba(245, 158, 11, 0.25) 0%, rgba(245, 158, 11, 0) 70%);
    border-radius: 50%;
  }
  .header-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #fde68a;
    margin-bottom: 1rem;
  }
  .header h1 {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    margin-bottom: 0.5rem;
  }
  .header p {
    color: #94a3b8;
    font-size: 0.975rem;
    max-width: 650px;
  }
  .header-meta {
    margin-top: 1.25rem;
    display: flex;
    flex-wrap: wrap;
    gap: 1.25rem;
    font-size: 0.825rem;
    color: #cbd5e1;
    border-top: 1px solid rgba(255, 255, 255, 0.12);
    padding-top: 1rem;
  }
  .header-meta span {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
  }

  /* Card Styles */
  .card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 1.75rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.04);
    margin-bottom: 1.75rem;
  }
  .card-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-main);
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
  }

  /* Sample Presets */
  .presets-bar {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1.25rem;
    padding-bottom: 1rem;
    border-bottom: 1px dashed var(--border-color);
  }
  .presets-label {
    font-size: 0.825rem;
    font-weight: 600;
    color: var(--text-muted);
    margin-right: 0.25rem;
  }
  .preset-btn {
    background: var(--bg-page);
    border: 1px solid var(--border-color);
    color: #334155;
    padding: 0.35rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 600;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .preset-btn:hover {
    background: var(--primary-light);
    color: var(--primary);
    border-color: #bae6fd;
    transform: translateY(-1px);
  }

  /* Form Controls */
  label {
    display: block;
    margin: 0.85rem 0 0.35rem;
    font-weight: 600;
    font-size: 0.9rem;
    color: #1e293b;
  }
  input[type=text], textarea {
    width: 100%;
    padding: 0.75rem 1rem;
    font-size: 0.95rem;
    font-family: inherit;
    color: var(--text-main);
    background: #ffffff;
    border: 1.5px solid var(--border-color);
    border-radius: 10px;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  input[type=text]:focus, textarea:focus {
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.15);
  }
  textarea {
    min-height: 140px;
    resize: vertical;
    line-height: 1.6;
  }
  .char-hint {
    display: flex;
    justify-content: flex-end;
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.35rem;
  }

  /* Button Actions */
  .actions {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-top: 1.25rem;
  }
  button[type=submit] {
    background: var(--primary);
    color: #ffffff;
    border: none;
    padding: 0.75rem 1.6rem;
    font-size: 0.95rem;
    font-weight: 600;
    border-radius: 10px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25);
    transition: background 0.2s, transform 0.1s;
  }
  button[type=submit]:hover:not(:disabled) {
    background: var(--primary-hover);
    transform: translateY(-1px);
  }
  button[type=submit]:disabled {
    opacity: 0.65;
    cursor: wait;
  }
  .btn-secondary {
    background: transparent;
    border: 1px solid var(--border-color);
    color: var(--text-muted);
    padding: 0.75rem 1.25rem;
    font-size: 0.95rem;
    font-weight: 600;
    border-radius: 10px;
    cursor: pointer;
    transition: background 0.2s, color 0.2s;
  }
  .btn-secondary:hover {
    background: #f1f5f9;
    color: var(--text-main);
  }

  /* Spinner & Error */
  #spinner {
    display: none;
    margin-top: 1.25rem;
    padding: 0.85rem 1rem;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    color: #166534;
    font-size: 0.9rem;
    font-weight: 600;
    display: none;
    align-items: center;
    gap: 0.65rem;
  }
  .spin-icon {
    width: 18px;
    height: 18px;
    border: 2.5px solid #86efac;
    border-top-color: #166534;
    border-radius: 50%;
    animation: spin 0.75s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  #errorBanner {
    display: none;
    margin-bottom: 1.5rem;
    padding: 1rem 1.25rem;
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #991b1b;
    border-radius: 10px;
    font-size: 0.9rem;
    font-weight: 500;
  }

  /* Results Section */
  #results {
    display: none;
    animation: fadeIn 0.3s ease-out;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

  /* Stats Grid */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
  }
  .stat-card {
    background: var(--bg-page);
    border: 1px solid var(--border-color);
    padding: 1rem 1.25rem;
    border-radius: 12px;
  }
  .stat-label {
    font-size: 0.775rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .stat-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--text-main);
    margin-top: 0.25rem;
  }

  /* Legend Chips */
  .legend-box {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
    margin: 1rem 0 1.25rem;
    padding: 0.85rem 1rem;
    background: var(--bg-page);
    border: 1px solid var(--border-color);
    border-radius: 10px;
  }
  .legend-title {
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--text-muted);
    margin-right: 0.25rem;
  }
  .legend-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    font-size: 0.775rem;
    font-weight: 700;
  }

  /* Annotated Text Display */
  #annotated {
    margin-top: 0.5rem;
    padding: 1.5rem;
    background: #ffffff;
    border: 1.5px solid var(--border-color);
    border-radius: 12px;
    white-space: pre-wrap;
    line-height: 2.3;
    font-size: 1.05rem;
    font-family: inherit;
    color: #1e293b;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.02);
  }
  mark {
    padding: 0.25em 0.5em;
    border-radius: 6px;
    font-weight: 600;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    transition: transform 0.1s, box-shadow 0.1s;
    cursor: default;
  }
  mark:hover {
    box-shadow: 0 4px 8px rgba(0,0,0,0.12);
  }
  .lbl {
    font-size: 0.7em;
    font-weight: 800;
    border-radius: 4px;
    padding: 0.15em 0.45em;
    margin-left: 0.4em;
    letter-spacing: 0.03em;
    vertical-align: baseline;
    opacity: 0.9;
  }

  /* Entities Table */
  .table-responsive {
    overflow-x: auto;
    margin-top: 0.75rem;
    border: 1px solid var(--border-color);
    border-radius: 10px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    background: #ffffff;
  }
  th {
    background: #f8fafc;
    color: #475569;
    font-weight: 700;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.75rem 1rem;
    border-bottom: 1.5px solid var(--border-color);
    text-align: left;
  }
  td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border-color);
    color: #334155;
  }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background-color: #f8fafc; }
  .entity-text-cell {
    font-weight: 600;
    color: var(--text-main);
  }
  .offset-cell {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.825rem;
    color: #64748b;
  }

  /* Tag styles per label */
  .badge-ORG    { background: var(--c-org-bg); color: var(--c-org-txt); border: 1px solid var(--c-org-border); }
  .badge-PERSON { background: var(--c-person-bg); color: var(--c-person-txt); border: 1px solid var(--c-person-border); }
  .badge-GPE    { background: var(--c-gpe-bg); color: var(--c-gpe-txt); border: 1px solid var(--c-gpe-border); }
  .badge-DATE   { background: var(--c-date-bg); color: var(--c-date-txt); border: 1px solid var(--c-date-border); }
  .badge-EVENT  { background: var(--c-event-bg); color: var(--c-event-txt); border: 1px solid var(--c-event-border); }
  .badge-MONEY  { background: var(--c-money-bg); color: var(--c-money-txt); border: 1px solid var(--c-money-border); }

  /* Utility & Footer */
  .copy-btn {
    background: #f1f5f9;
    border: 1px solid var(--border-color);
    color: #475569;
    padding: 0.35rem 0.75rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }
  .copy-btn:hover { background: #e2e8f0; color: #0f172a; }
  .footer {
    text-align: center;
    margin-top: 3rem;
    font-size: 0.825rem;
    color: var(--text-muted);
  }
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <header class="header">
    <h1>Maybank News NER Extraction</h1>
    <p>Layanan ekstraksi otomatis entitas ORG, PERSON, GPE, DATE, EVENT, dan MONEY dari judul serta isi berita perbankan berbasis spaCy fine-tuned.</p>
  </header>

  <!-- Error Banner -->
  <div id="errorBanner" role="alert"></div>

  <!-- Form Card -->
  <div class="card">
    <div class="card-title">
      <span>Input Berita</span>
    </div>

    <!-- Quick Presets -->
    <div class="presets-bar">
      <span class="presets-label">Coba Contoh Cepat:</span>
      <button type="button" class="preset-btn" onclick="loadSample(1)">Laba Maybank (Finance)</button>
      <button type="button" class="preset-btn" onclick="loadSample(2)">Putusan Pengadilan (Legal)</button>
      <button type="button" class="preset-btn" onclick="loadSample(3)">Golf Championship (Sports)</button>
      <button type="button" class="preset-btn" onclick="clearForm()" style="margin-left:auto; color:#ef4444; border-color:#fecaca;">Kosongkan</button>
    </div>

    <form id="nerForm">
      <label for="title">Judul Berita (Title)</label>
      <input type="text" id="title" name="title" placeholder="Masukkan judul berita, contoh: Maybank 2Q profit rises">
      
      <label for="body">Isi Berita (Body)</label>
      <textarea id="body" name="body" placeholder="Tempel atau ketik teks artikel berita di sini..." oninput="updateCharCount()"></textarea>
      <div class="char-hint">
        <span id="charCount">0</span>&nbsp;/ 100.000 karakter
      </div>

      <div class="actions">
        <button type="submit" id="submitBtn">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
          Ekstraksi Entitas
        </button>
        <button type="button" class="btn-secondary" onclick="clearForm()">Reset</button>
      </div>
    </form>

    <div id="spinner" aria-live="polite">
      <div class="spin-icon"></div>
      <span>Sedang memproses teks dengan model spaCy...</span>
    </div>
  </div>

  <!-- Results Section -->
  <div id="results">
    
    <!-- Stats Cards -->
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">Total Entitas Ditemukan</div>
        <div class="stat-value" id="count">0</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Total Karakter Dianalisis</div>
        <div class="stat-value" id="totalChars">0</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Model Engine</div>
        <div class="stat-value" style="font-size:1.15rem; color:#0284c7; padding-top:0.4rem;">maybank-ner-1.0</div>
      </div>
    </div>

    <!-- Annotated Text Card -->
    <div class="card">
      <div class="card-title">
        <span>Teks Beranotasi (Highlight Visual)</span>
        <button class="copy-btn" onclick="copyAnalyzedText()">Salin Teks</button>
      </div>
      
      <!-- Legend Chips -->
      <div class="legend-box">
        <span class="legend-title">Keterangan Label:</span>
        <span class="legend-chip badge-ORG">ORG &bull; Organisasi</span>
        <span class="legend-chip badge-PERSON">PERSON &bull; Orang</span>
        <span class="legend-chip badge-GPE">GPE &bull; Lokasi/Negara</span>
        <span class="legend-chip badge-DATE">DATE &bull; Tanggal/Waktu</span>
        <span class="legend-chip badge-EVENT">EVENT &bull; Acara</span>
        <span class="legend-chip badge-MONEY">MONEY &bull; Uang</span>
      </div>

      <div id="annotated"></div>
    </div>

    <!-- Table Card -->
    <div class="card">
      <div class="card-title">
        <span>Tabel Rincian Entitas</span>
        <button class="copy-btn" onclick="copyJsonPayload()">Salin JSON API</button>
      </div>
      <div class="table-responsive">
        <table>
          <thead>
            <tr>
              <th style="width: 50px;">#</th>
              <th>Teks Entitas</th>
              <th>Tipe Label</th>
              <th>Start Char</th>
              <th>End Char</th>
              <th>Panjang</th>
            </tr>
          </thead>
          <tbody id="rows"></tbody>
        </table>
      </div>
    </div>

  </div>

  <footer class="footer">
    <p>&copy; 2026 Coursework Data Science — Named Entity Recognition Web Service</p>
  </footer>

</div>

<script>
let lastPayload = null;

// Presets
const SAMPLES = {
  1: {
    title: "Maybank 2Q profit rises",
    body: "Maybank posted a net profit of RM2.69 billion in Kuala Lumpur on Wednesday."
  },
  2: {
    title: "High Court to rule Oct 23 on Maritime Network bid",
    body: "The High Court here today fixed Oct 23 to rule on a stay application by Maritime Network Sdn Bhd in Kuala Lumpur."
  },
  3: {
    title: "Maybank Championship 2026 kicks off in Seri Kembangan",
    body: "The prestigious golf tournament Maybank Championship will take place in Seri Kembangan on 1 Oktober, featuring professional golfer Kelly Tan."
  }
};

function loadSample(id) {
  if (SAMPLES[id]) {
    document.getElementById("title").value = SAMPLES[id].title;
    document.getElementById("body").value = SAMPLES[id].body;
    updateCharCount();
    hideError();
  }
}

function clearForm() {
  document.getElementById("title").value = "";
  document.getElementById("body").value = "";
  updateCharCount();
  hideError();
  document.getElementById("results").style.display = "none";
}

function updateCharCount() {
  const title = document.getElementById("title").value.trim();
  const body = document.getElementById("body").value.trim();
  const text = title ? (title + (body ? "\n\n" + body : "")) : body;
  document.getElementById("charCount").textContent = text.length.toLocaleString();
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function showError(msg) {
  const banner = document.getElementById("errorBanner");
  banner.textContent = msg;
  banner.style.display = "block";
}

function hideError() {
  const banner = document.getElementById("errorBanner");
  banner.style.display = "none";
  banner.textContent = "";
}

// Label Styles Helper
const LABEL_CLASSES = {
  "ORG": "badge-ORG",
  "PERSON": "badge-PERSON",
  "GPE": "badge-GPE",
  "DATE": "badge-DATE",
  "EVENT": "badge-EVENT",
  "MONEY": "badge-MONEY"
};

function getClassForLabel(label) {
  return LABEL_CLASSES[label] || "badge-ORG";
}

// Form Submit
document.getElementById("nerForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();
  const btn = document.getElementById("submitBtn");
  const spinner = document.getElementById("spinner");
  const results = document.getElementById("results");

  btn.disabled = true;
  spinner.style.display = "flex";

  try {
    const resp = await fetch("/extract-entities", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        title: document.getElementById("title").value,
        body: document.getElementById("body").value
      })
    });
    const payload = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      showError(payload.error || ("Permintaan gagal dengan status " + resp.status));
      results.style.display = "none";
      return;
    }
    lastPayload = payload;
    renderResults(payload);
  } catch (err) {
    showError("Network error: " + err.message);
  } finally {
    btn.disabled = false;
    spinner.style.display = "none";
  }
});

function renderResults(payload) {
  const entities = payload.entities || [];
  document.getElementById("count").textContent = entities.length;
  
  const rawText = document.getElementById("title").value.trim() + "\n\n" + document.getElementById("body").value.trim();
  document.getElementById("totalChars").textContent = rawText.length.toLocaleString();

  // Render Table Rows
  const rows = document.getElementById("rows");
  rows.innerHTML = "";
  if (entities.length === 0) {
    const tr = document.createElement("tr");
    tr.innerHTML = '<td colspan="6" style="text-align:center; color:#94a3b8; padding:2rem;">Tidak ada entitas yang terdeteksi pada teks ini.</td>';
    rows.appendChild(tr);
  } else {
    entities.forEach((e, idx) => {
      const tr = document.createElement("tr");
      const badgeCls = getClassForLabel(e.label);
      const spanLen = e.end - e.start;
      tr.innerHTML = `
        <td style="color:#94a3b8; font-weight:600;">${idx + 1}</td>
        <td class="entity-text-cell">${escapeHtml(e.text)}</td>
        <td><span class="legend-chip ${badgeCls}">${escapeHtml(e.label)}</span></td>
        <td class="offset-cell">${e.start}</td>
        <td class="offset-cell">${e.end}</td>
        <td class="offset-cell">${spanLen} chars</td>
      `;
      rows.appendChild(tr);
    });
  }

  // Render Annotated Text
  renderAnnotated(entities, rawText);
  
  const resultsCard = document.getElementById("results");
  resultsCard.style.display = "block";
  resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderAnnotated(entities, raw) {
  const box = document.getElementById("annotated");
  const sorted = [...entities].sort((a, b) => a.start - b.start || a.end - b.end);
  let out = "";
  let pos = 0;

  for (const e of sorted) {
    if (e.start < pos) continue; // skip overlapping span
    out += escapeHtml(raw.slice(pos, e.start));
    const badgeCls = getClassForLabel(e.label);
    out += `<mark class="${badgeCls}">${escapeHtml(raw.slice(e.start, e.end))}<span class="lbl ${badgeCls}">${escapeHtml(e.label)}</span></mark>`;
    pos = e.end;
  }
  out += escapeHtml(raw.slice(pos));
  box.innerHTML = out;
}

function copyAnalyzedText() {
  const text = document.getElementById("annotated").innerText;
  navigator.clipboard.writeText(text).then(() => {
    alert("Teks berhasil disalin ke clipboard!");
  });
}

function copyJsonPayload() {
  if (!lastPayload) return;
  navigator.clipboard.writeText(JSON.stringify(lastPayload, null, 2)).then(() => {
    alert("JSON respons berhasil disalin!");
  });
}

// Initial Char Count
updateCharCount();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    # Local/dev entry point. In production, run behind a WSGI server.
    app.run(host="127.0.0.1", port=5000)
