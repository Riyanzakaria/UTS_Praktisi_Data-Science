"""API tests for the Maybank NER Flask service.

Role 3 (Service & Docs Engineer) coursework deliverable. These tests run
against the REAL trained pipeline (``models/ner_model``) via Flask's
``test_client`` -- no mocking of the model.
"""

import json
import os
import sys

try:
    import pytest
except ImportError:
    pytest = None

# Make the project root importable so `src.app` resolves when pytest is run
# from the project directory (or anywhere else).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import app as app_module  # noqa: E402

MAX_INPUT_CHARS = app_module.MAX_INPUT_CHARS

FIXED_TITLE = "Maybank 2Q profit rises"
FIXED_BODY = (
    "Maybank posted a net profit of RM2.69 billion in Kuala Lumpur on Wednesday."
)


if pytest is not None:
    @pytest.fixture(scope="module")
    def client():
        """Flask test client sharing the once-loaded model across tests."""
        app_module.app.config["TESTING"] = True
        with app_module.app.test_client() as c:
            yield c
else:
    def client():
        pass


def combined_text(title: str, body: str) -> str:
    return title.strip() + "\n\n" + body.strip()


def post_extract(client, payload, raw=None, content_type="application/json"):
    if raw is not None:
        return client.post(
            "/extract-entities", data=raw, content_type=content_type
        )
    return client.post("/extract-entities", json=payload)


# ---------------------------------------------------------------------------
# (a) both-empty -> 400
# ---------------------------------------------------------------------------

def test_both_empty_returns_400(client):
    resp = post_extract(client, {"title": "", "body": ""})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_whitespace_only_returns_400(client):
    resp = post_extract(client, {"title": "   ", "body": "\n\t "})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# (b) malformed JSON -> 400
# ---------------------------------------------------------------------------

def test_malformed_json_returns_400(client):
    resp = post_extract(client, None, raw="{not valid json")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_non_object_json_returns_400(client):
    resp = post_extract(client, None, raw="[1, 2, 3]")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# (c) non-string title -> 400
# ---------------------------------------------------------------------------

def test_non_string_title_returns_400(client):
    resp = post_extract(client, {"title": 123, "body": "some text"})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_missing_body_returns_400(client):
    resp = post_extract(client, {"title": "a title"})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# (d) input > 100000 chars -> 400
# ---------------------------------------------------------------------------

def test_input_over_limit_returns_400(client):
    title = "a" * (MAX_INPUT_CHARS + 1)  # combined text will exceed the limit
    resp = post_extract(client, {"title": title, "body": ""})
    assert resp.status_code == 400
    assert "too long" in resp.get_json()["error"]


# ---------------------------------------------------------------------------
# (e) valid request -> 200, offsets slice back exactly, fixed KL sample
# ---------------------------------------------------------------------------

def test_valid_request_offsets_and_kuala_lumpur(client):
    resp = post_extract(client, {"title": FIXED_TITLE, "body": FIXED_BODY})
    assert resp.status_code == 200

    payload = resp.get_json()
    assert "entities" in payload and "meta" in payload
    assert payload["meta"]["model"] == app_module.MODEL_NAME
    assert payload["meta"]["entity_count"] == len(payload["entities"])

    # Offsets must be char offsets into title.strip() + "\n\n" + body.strip()
    text = combined_text(FIXED_TITLE, FIXED_BODY)
    for ent in payload["entities"]:
        assert text[ent["start"]:ent["end"]] == ent["text"], (
            f"offset slice mismatch for {ent!r}"
        )
        assert isinstance(ent["label"], str) and ent["label"]

    # Stable, model-robust check: the fixed sample must yield a GPE
    # "Kuala Lumpur" (deliberately not hard-coding any other spans).
    gpes = [e for e in payload["entities"] if e["label"] == "GPE"]
    assert any(e["text"] == "Kuala Lumpur" for e in gpes), (
        f"expected a GPE 'Kuala Lumpur', got: {gpes}"
    )


def test_entities_sorted_and_deduplicated(client):
    resp = post_extract(client, {"title": FIXED_TITLE, "body": FIXED_BODY})
    entities = resp.get_json()["entities"]
    starts = [(e["start"], e["end"]) for e in entities]
    assert starts == sorted(starts)
    assert len({(e["start"], e["end"], e["label"], e["text"]) for e in entities}) == len(
        entities
    )


# ---------------------------------------------------------------------------
# (f) determinism: two identical calls return byte-identical JSON
# ---------------------------------------------------------------------------

def test_identical_requests_are_deterministic(client):
    payload = {"title": FIXED_TITLE, "body": FIXED_BODY}
    first = post_extract(client, payload).data
    second = post_extract(client, payload).data
    assert first == second


def test_deterministic_response_matches_canonical_form(client):
    resp = post_extract(client, {"title": FIXED_TITLE, "body": FIXED_BODY})
    canonical = json.dumps(resp.get_json(), separators=(",", ":")).encode()
    # Flask's jsonify is already compact and deterministic; the raw body must
    # parse to the same object we canonicalize here.
    assert json.loads(resp.data) == json.loads(canonical)


# ---------------------------------------------------------------------------
# panel smoke test
# ---------------------------------------------------------------------------

def test_index_panel_contains_markup(client):
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    for marker in ("nerForm", "errorBanner", "results", "annotated"):
        assert f'id="{marker}"' in html
    assert "<table>" in html


if __name__ == "__main__":
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        test_funcs = [
            test_both_empty_returns_400,
            test_whitespace_only_returns_400,
            test_malformed_json_returns_400,
            test_non_object_json_returns_400,
            test_non_string_title_returns_400,
            test_missing_body_returns_400,
            test_input_over_limit_returns_400,
            test_valid_request_offsets_and_kuala_lumpur,
            test_entities_sorted_and_deduplicated,
            test_identical_requests_are_deterministic,
            test_deterministic_response_matches_canonical_form,
            test_index_panel_contains_markup,
        ]
        passed = 0
        for fn in test_funcs:
            fn(c)
            print(f"PASSED {fn.__name__}")
            passed += 1
        print(f"\n==================== {passed}/{len(test_funcs)} passed ====================")
