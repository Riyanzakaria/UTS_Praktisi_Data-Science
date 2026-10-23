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
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Maybank NER — Entity Extraction</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
  label { display: block; margin: 1rem 0 0.25rem; font-weight: 600; }
  input[type=text], textarea { width: 100%; padding: .5rem; font-size: 1rem; }
  textarea { min-height: 160px; font-family: inherit; }
  button { margin-top: 1rem; padding: .6rem 1.4rem; font-size: 1rem; cursor: pointer; }
  button:disabled { opacity: .5; cursor: wait; }
  #spinner { display: none; margin-top: 1rem; color: #555; }
  #errorBanner { display: none; margin-top: 1rem; padding: .75rem; background: #fdecea; border: 1px solid #f5c6cb; color: #a4262c; border-radius: 4px; }
  #results { display: none; margin-top: 1.5rem; }
  table { width: 100%; border-collapse: collapse; margin-top: .5rem; }
  th, td { border: 1px solid #ccc; padding: .4rem .6rem; text-align: left; }
  th { background: #f2f2f2; }
  #annotated { margin-top: 1rem; padding: .75rem; border: 1px solid #ddd; border-radius: 4px; white-space: pre-wrap; line-height: 1.9; }
  mark { padding: .1em .3em; border-radius: 3px; }
  .lbl { font-size: .7em; font-weight: 700; color: #fff; background: rgba(0,0,0,.45); border-radius: 3px; padding: .05em .3em; margin-left: .25em; vertical-align: super; }
</style>
</head>
<body>
<h1>Entity Extraction</h1>
<div id="errorBanner" role="alert"></div>
<form id="nerForm">
  <label for="title">Title</label>
  <input type="text" id="title" name="title" placeholder="e.g. Maybank Q3 results">
  <label for="body">Body</label>
  <textarea id="body" name="body" placeholder="Paste the text to analyze..."></textarea>
  <button type="submit" id="submitBtn">Submit</button>
</form>
<div id="spinner" aria-live="polite">Analyzing&hellip; please wait.</div>
<div id="results">
  <h2>Results (<span id="count">0</span>)</h2>
  <table>
    <thead><tr><th>Text</th><th>Label</th><th>Start</th><th>End</th></tr></thead>
    <tbody id="rows"></tbody>
  </table>
  <h2>Annotated text</h2>
  <div id="annotated"></div>
</div>
<script>
const LABEL_COLORS = {};  // label -> color, assigned on the fly
const PALETTE = ["#ffe082", "#a5d6a7", "#90caf9", "#ce93d8", "#ffab91", "#80cbc4", "#f48fb1", "#fff59d"];
let paletteIdx = 0;

function colorFor(label) {
  if (!(label in LABEL_COLORS)) {
    LABEL_COLORS[label] = PALETTE[paletteIdx % PALETTE.length];
    paletteIdx += 1;
  }
  return LABEL_COLORS[label];
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

document.getElementById("nerForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();
  const btn = document.getElementById("submitBtn");
  const spinner = document.getElementById("spinner");
  const results = document.getElementById("results");
  btn.disabled = true;
  spinner.style.display = "block";

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
      showError(payload.error || ("Request failed with status " + resp.status));
      results.style.display = "none";
      return;
    }
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
  const rows = document.getElementById("rows");
  rows.innerHTML = "";
  for (const e of entities) {
    const tr = document.createElement("tr");
    for (const k of ["text", "label", "start", "end"]) {
      const td = document.createElement("td");
      td.textContent = e[k];
      tr.appendChild(td);
    }
    rows.appendChild(tr);
  }
  renderAnnotated(entities);
  document.getElementById("results").style.display = "block";
}

function renderAnnotated(entities) {
  // Rebuild the analyzed text client-side: title.strip() + "\n\n" + body.strip()
  const raw = document.getElementById("title").value.trim() + "\n\n" + document.getElementById("body").value.trim();
  const box = document.getElementById("annotated");
  const sorted = [...entities].sort((a, b) => a.start - b.start || a.end - b.end);
  let out = "";
  let pos = 0;
  for (const e of sorted) {
    if (e.start < pos) continue;  // skip overlapping span
    out += escapeHtml(raw.slice(pos, e.start));
    const color = colorFor(e.label);
    out += '<mark style="background:' + color + '">' + escapeHtml(raw.slice(e.start, e.end)) +
           '<span class="lbl">' + escapeHtml(e.label) + "</span></mark>";
    pos = e.end;
  }
  out += escapeHtml(raw.slice(pos));
  box.innerHTML = out;
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    # Local/dev entry point. In production, run behind a WSGI server.
    app.run(host="127.0.0.1", port=5000)
