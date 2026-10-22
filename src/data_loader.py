#!/usr/bin/env python3
"""Data loading for the NER coursework project.

Loads the 8 Maybank online-media clipping CSVs (semicolon-delimited, with a
single leading single-quote Excel artifact on many cells), dedupes articles by
their cleaned ``original_id`` (first occurrence wins), and builds the
annotation text via the deterministic rule used in
``~/workspace/study-case/selection_manifest.json``.

Running ``python -m src.data_loader`` (from the project root) or the file
directly prints the dataset profile and writes it to
``project/data/profile_report.json``.
"""
import glob
import json
import os
import re
from typing import Dict, List

import pandas as pd

# CSV files live OUTSIDE the project tree (never copied into project/).
BASE = os.path.expanduser('~/workspace/study-case')
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# Sentence terminator candidates: one of . ! ? followed by whitespace or end.
_SENT_END = re.compile(r'[.!?](?=\s|$)')


def clean(s) -> str:
    """Strip whitespace and ONE leading single-quote Excel artifact.

    Returns "" for None. The second strip handles cells like ``"'en"``
    where removing the quote exposes more whitespace.
    """
    s = (s or '').strip()
    if s.startswith("'"):
        s = s[1:]
    return s.strip()


def make_text(title, body) -> str:
    """Build the annotation text: clean title + "\\n\\n" + body excerpt.

    Excerpt rule (identical to selection_manifest.json): take the first 600
    chars of the cleaned body. If the body is longer than 600 chars, cut back
    to the last sentence terminator (. ! ? followed by whitespace/end) whose
    position is >= 300; if none exists, hard-cut at 600. The result is
    right-stripped.
    """
    t, b = clean(title), clean(body)
    ex = b[:600]
    if len(b) > 600:
        m = None
        for mm in _SENT_END.finditer(b[:600]):
            if mm.start() >= 300:
                m = mm
        if m:
            ex = b[:m.end()]
        ex = ex.rstrip()
    return f'{t}\n\n{ex}'


def load_articles() -> List[Dict]:
    """Load all 8 CSVs and dedupe by cleaned original_id (first wins).

    Returns a list of dicts with keys: article_id, source_file, row_index,
    source_name, title, body, date_published, language. The number of rows
    dropped as duplicates is attached as ``load_articles.duplicates_dropped``.
    """
    files = sorted(glob.glob(os.path.join(BASE, '*.csv')))
    rows: List[Dict] = []
    seen = set()
    duplicates = 0
    for f in files:
        df = pd.read_csv(f, sep=';', dtype=str, keep_default_na=False,
                         encoding='utf-8-sig')
        basename = os.path.basename(f)
        for idx, r in df.iterrows():
            oid = clean(r.get('original_id', ''))
            if oid in seen:
                duplicates += 1
                continue
            seen.add(oid)
            rows.append({
                'article_id': oid,
                'source_file': basename,
                'row_index': int(idx),
                'source_name': clean(r.get('source_name', '')),
                'title': clean(r.get('title', '')),
                'body': clean(r.get('body', '')),
                'date_published': clean(r.get('date_published', '')),
                'language': clean(r.get('language', '')),
            })
    load_articles.duplicates_dropped = duplicates
    load_articles.file_count = len(files)
    load_articles.rows_per_file = {
        os.path.basename(f): len(pd.read_csv(f, sep=';', dtype=str,
                                             keep_default_na=False,
                                             encoding='utf-8-sig'))
        for f in files
    }
    return rows


def profile() -> Dict:
    """Profile the dataset: counts, missingness, languages, entities column.

    Prints the profile and writes it to ``project/data/profile_report.json``.
    """
    files = sorted(glob.glob(os.path.join(BASE, '*.csv')))
    rows_per_file = {}
    total_rows = 0
    lang_counts: Dict[str, int] = {}
    missing_title = 0
    missing_body = 0
    entities_empty = 0  # rows whose 'entities' cell is empty after cleaning
    seen = set()
    duplicates = 0
    for f in files:
        df = pd.read_csv(f, sep=';', dtype=str, keep_default_na=False,
                         encoding='utf-8-sig')
        rows_per_file[os.path.basename(f)] = len(df)
        total_rows += len(df)
        for _, r in df.iterrows():
            oid = clean(r.get('original_id', ''))
            if oid in seen:
                duplicates += 1
                continue
            seen.add(oid)
            if not clean(r.get('title', '')):
                missing_title += 1
            if not clean(r.get('body', '')):
                missing_body += 1
            lang = clean(r.get('language', ''))
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
            if not clean(r.get('entities', '')):
                entities_empty += 1
    unique_rows = len(seen)
    report = {
        'rows_per_file': rows_per_file,
        'total_rows': total_rows,
        'unique_rows': unique_rows,
        'duplicate_count': duplicates,
        'missing_title': missing_title,
        'missing_body': missing_body,
        'language_distribution': lang_counts,
        'entities_empty_count': entities_empty,
        'english_unique_count': lang_counts.get('en', 0),
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, 'profile_report.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return report


def main():
    report = profile()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
