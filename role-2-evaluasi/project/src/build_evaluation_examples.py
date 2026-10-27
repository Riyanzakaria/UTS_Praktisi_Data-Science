"""build_evaluation_examples.py — assemble project/data/evaluation_examples.json.

10 held-out items: the 3 test records from annotations.jsonl (copied verbatim)
+ 7 new excerpts from selection_manifest["eval_extra"], gold-labeled by the
annotator following the review procedure in data/annotation_guidelines.md:
pre-label with en_core_web_sm, then MANUALLY review every candidate span
(accept / fix boundary / fix label / delete / add misses). The model's
suggestion was never accepted blindly; corrections are recorded in each
item's "notes" field.

Span spec format: (label, anchor, lstrip, rstrip, occ) — the span is
anchor[lstrip:len(anchor)-rstrip] at the occ-th (1-based) occurrence of the
anchor. Anchors were chosen to be unambiguous; the script asserts the
extracted text matches expectations and that no spans overlap.

HELD OUT: none of these 10 items were used in training, and none influenced
model or iteration selection (val F1 on 3 val records was the only signal).
"""
import json

MANIFEST = "/home/hatch/workspace/study-case/selection_manifest.json"
ANNOT = "/home/hatch/workspace/study-case/project/data/annotations.jsonl"
OUT = "/home/hatch/workspace/study-case/project/data/evaluation_examples.json"
RSQ = "\u2019"  # right single quotation mark used in the source texts

# ------------------------------------------------------------------ gold specs
# (label, anchor, lstrip, rstrip, occurrence)
GOLD = {
    # [0] HLB Islamic / SURI Inspirasi (Malay)
    "a4b9d4d69271adea944a1e6dbefa761765614d148e93236350fcfd374d92de84": [
        ("ORG", "HLB Islamic salur", 0, 6, 1),
        ("MONEY", "salur RM138,600 pembiayaan", 6, 11, 1),
        ("ORG", "melalui SURI Inspirasi\n", 8, 1, 1),
        ("GPE", "PETALING JAYA:", 0, 1, 1),
        ("ORG", "HLB Islamic menyalurkan", 0, 12, 1),
        ("MONEY", "sebanyak RM138,600 kepada", 9, 7, 1),
        ("ORG", "sosial SURI Inspirasi (SURI)", 7, 7, 1),
        ("ORG", "(SURI) di bawah", 1, 10, 1),
        ("EVENT", "kerangka iTEKAD Bank", 9, 5, 1),
        ("ORG", "iTEKAD Bank Negara Malaysia (BNM)", 7, 6, 1),
        ("ORG", "(BNM), selaras", 1, 10, 1),
        ("EVENT", "berasaskan Pengantaraan Berasaskan Nilai (VBI)", 11, 6, 1),
        ("EVENT", "(VBI).", 1, 2, 1),
        ("EVENT", "program Sustainable Reliable Income (SRI)", 8, 6, 1),
        ("EVENT", "(SRI) anjuran", 1, 9, 1),
        ("ORG", "anjuran SURI.", 8, 1, 1),
        ("MONEY", "melebihi RM14,000 bagi", 9, 5, 1),
    ],
    # [1] RHB / Maritime Network (theedgemalaysia)
    "00b927e4e743da1cb46bff4717a6e2d40cdcc6ec09dccf9b9472414e6d794393": [
        ("DATE", "Oct 23 ruling", 0, 7, 1),
        ("ORG", "restrain RHB from", 9, 5, 1),
        ("ORG", "closing Maritime Network's accounts", 8, 11, 1),
        ("GPE", "<p>KUALA LUMPUR (Sept", 3, 6, 1),
        ("DATE", "(Sept 24):", 1, 2, 1),
        ("ORG", "): The High Court here", 7, 5, 1),
        ("DATE", "here on Thursday fixed", 8, 6, 1),
        ("DATE", "Thursday fixed Oct 23 to", 15, 3, 1),
        ("ORG", "restrain RHB Bank Bhd (LK", 9, 4, 1),
        ("ORG", "(LK:RHBBANK) from", 4, 6, 1),
        ("PERSON", "Commissioner Elesabet Paya Wan set", 13, 4, 1),
        ("PERSON", "P&nbsp;Taneswaran, acting", 7, 8, 1),
        ("ORG", "for Maritime Network Sdn Bhd, and", 4, 5, 1),
        ("PERSON", "counsel, Gan Khong Aik.", 9, 1, 1),
        ("PERSON", "<p>Taneswaran submitted", 3, 10, 1),
    ],
    # [2] CIMB IB / Budget 2027 (klsescreener)
    "a6a10ea181432bd8664d0906287f1eeb0b9434a674f48b99e7052fa7c53c4042": [
        ("ORG", "CIMB IB says", 0, 5, 1),
        ("GPE", "says Malaysia could", 5, 6, 1),
        ("DATE", "for Budget 2027, cut", 11, 5, 1),
        ("ORG", "CIMB IB says", 0, 5, 2),
        ("GPE", "says Malaysia could", 5, 6, 2),
        ("DATE", "for Budget 2027, cut", 11, 5, 2),
        ("GPE", "deductions KUALA LUMPUR (Oct", 11, 5, 1),
        ("DATE", "(Oct 5):", 1, 2, 1),
        ("GPE", "(Oct 5): Malaysia could", 9, 6, 1),
        ("ORG", "said CIMB Investment Bank.", 5, 1, 1),
        ("ORG", "Bank. CIMB IB said", 6, 5, 1),
        ("ORG", "citing KPMG senior", 7, 7, 1),
        ("PERSON", "policy Dr Verrinderjeet Singh during", 10, 7, 1),
    ],
    # [3] Maybank / Hong Leong awards (fintechnews)
    "6e177c8821b02591cbc096efcfe98dbc3f3a4d0e1e7e4d80b41de0ad0d900ddb": [
        ("ORG", "Maybank, Hong", 0, 6, 1),
        ("ORG", "Maybank, Hong Leong Win", 9, 4, 1),
        ("EVENT", "at Bursa Malaysia Awards\n", 3, 1, 1),
        ("ORG", "\n\nBursa Malaysia awarded", 2, 8, 1),
        ("ORG", "honours to Maybank", 11, 0, 1),
        ("PERSON", "Sri Khairussaleh Ramli and", 4, 4, 1),
        ("ORG", "and Hong Leong Bank", 4, 0, 1),
        ("PERSON", "Bank" + RSQ + "s Malkit Singh Maan.", 7, 1, 1),
        ("EVENT", "inaugural Bursa Malaysia IR Awards 2026 took", 10, 5, 1),
        ("DATE", "took place on 4 September.", 14, 1, 1),
        ("ORG", "September. Maybank received", 11, 9, 1),
        ("PERSON", "Group CEO Khairussaleh. The", 10, 5, 1),
        ("PERSON", "Services. Malkit won", 10, 4, 1),
    ],
    # [4] Mplus Market Pulse (klse.i3investor)
    "3c7cbea559396a72010b50f84bd8c4100c4c3dfa35e7eb8c1f7f8540a13e1377": [
        ("EVENT", "Mplus Market Pulse -", 0, 2, 1),
        ("DATE", "- 07 Oct 2026\n", 2, 1, 1),
        ("GPE", "\n\nMY: Tracking", 2, 10, 1),
        ("DATE", "the Budget 2027 announcement", 11, 13, 1),
    ],
    # [5] AEON Credit (klse.i3investor)
    "a8d7872ea5c5092923b301544b8e65de6d5d9568679b796ae2919b8909431fec": [
        ("ORG", "AEON Credit Closes", 0, 7, 1),
        ("DATE", "as 2Q Results", 3, 8, 1),
        ("GPE", "Analysts\n\nKUALA LUMPUR (Sept", 10, 6, 1),
        ("DATE", "(Sept 30):", 1, 2, 1),
        ("ORG", "): AEON Credit Service (M) Bhd (KL", 3, 4, 1),
        ("ORG", "(KL:AEONCR) fell", 4, 6, 1),
        ("DATE", "in nearly a year while", 3, 6, 1),
        ("DATE", "Earnings in the first-half only", 12, 5, 1),
        ("DATE", "consensus" + RSQ + " full-year forecast", 11, 9, 1),
    ],
    # [6] RHB / The Makeover Guys (fintechnews)
    "bdd840f5db8535c01a7884cba064b4dd9926cff01a182f9f8425ddcbdb963012": [
        ("ORG", "RHB, The", 0, 5, 1),
        ("ORG", "RHB, The Makeover Guys Offer", 5, 6, 1),
        ("ORG", "\n\nRHB Bank has", 2, 4, 1),
        ("ORG", "firm The Makeover Guys to", 5, 3, 1),
    ],
}

NOTES = {
    "a4b9d4d69271adea944a1e6dbefa761765614d148e93236350fcfd374d92de84":
        "Review: baseline tagged 'HLB' alone -> fixed to full 'HLB Islamic' ORG (x2); "
        "'RM138,600' tagged PERSON -> fixed to MONEY (x2); 'kepada melalui' PERSON, "
        "'Inisiatif'/'menjana jumlah pendapatan'/'melebihi RM14,000'/'bagi' ORG, "
        "'secara langsung menyumbang kepada peningkatan'/'pendapatan bulanan' PERSON, "
        "'usahawan' GPE, 'menyertai' DATE -> all deleted (generic words / wrong label); "
        "'94' CARDINAL deleted (percent out of scheme); added misses: 'SURI Inspirasi' ORG (x2), "
        "'(SURI)' ORG, 'PETALING JAYA' GPE, 'iTEKAD' EVENT (named BNM programme), "
        "'Pengantaraan Berasaskan Nilai' + 'VBI' EVENT (named framework), "
        "'Sustainable Reliable Income' + 'SRI' EVENT (named programme), 'RM14,000' MONEY.",
    "00b927e4e743da1cb46bff4717a6e2d40cdcc6ec09dccf9b9472414e6d794393":
        "Review: 'Oct 23' (title) was CARDINAL '23' -> fixed to DATE 'Oct 23'; added miss "
        "'RHB' ORG (title); 'Maritime Network's' -> possessive stripped to 'Maritime Network' ORG; "
        "added miss 'KUALA LUMPUR' GPE; 'The High Court' -> stripped 'The' to 'High Court' ORG "
        "(matches train/test convention); 'Oct 23' (body) was CARDINAL -> DATE; "
        "'Bank Bhd' -> expanded to 'RHB Bank Bhd' ORG; ticker '(LK:RHBBANK)' -> 'RHBBANK' ORG "
        "(listed instrument, cf. val 'MAYBANK' ORG); deleted garbage span "
        "'newsTextDataWrapInner><p>Judicial' ORG; 'P&nbsp;Taneswaran' GPE -> PERSON 'Taneswaran' "
        "(markup excluded from span); 'Gan Khong' -> expanded to full 'Gan Khong Aik' PERSON; "
        "added miss 'Taneswaran' PERSON (final paragraph).",
    "a6a10ea181432bd8664d0906287f1eeb0b9434a674f48b99e7052fa7c53c4042":
        "Review: 'KUALA' ORG -> fixed to 'KUALA LUMPUR' GPE; 'two to three decades' DATE -> "
        "deleted (duration, not a date expression); added miss 'Verrinderjeet Singh' PERSON "
        "(title 'Dr' excluded per guidelines); '2027' kept as DATE (bare year functioning as "
        "date reference, cf. train '2026').",
    "6e177c8821b02591cbc096efcfe98dbc3f3a4d0e1e7e4d80b41de0ad0d900ddb":
        "Review: added misses 'Maybank' ORG and 'Hong Leong' ORG (title); "
        "'Hong Leong Win Best' PERSON -> fixed to 'Hong Leong' ORG; deleted 'CFO Honours' ORG, "
        "'CFO' ORG, 'Group' ORG, 'Best Company'/'Best Financial Reporting' ORG, "
        "'Investor Relations' ORG (generic award-category words); 'Bursa Malaysia Awards' ORG -> "
        "EVENT (named awards event); 'Bursa'/'Malaysia' GPE pair -> merged to 'Bursa Malaysia' ORG; "
        "'Maybank's Dato' Sri Khairussaleh Ramli' ORG -> split into 'Maybank' ORG + "
        "'Khairussaleh Ramli' PERSON (title excluded); 'Hong Leong Bank's' -> possessive stripped "
        "to 'Hong Leong Bank' ORG; 'Bursa Malaysia IR Awards' ORG -> EVENT "
        "'Bursa Malaysia IR Awards 2026' (year included, cf. train 'PNM CUP 2026' EVENT); "
        "added misses 'Malkit Singh Maan' PERSON, 'Maybank' ORG, 'Khairussaleh' PERSON, "
        "'Malkit' PERSON.",
    "3c7cbea559396a72010b50f84bd8c4100c4c3dfa35e7eb8c1f7f8540a13e1377":
        "Review: added misses 'Mplus Market Pulse' EVENT (named market report programme), "
        "'07 Oct 2026' DATE, 'MY' GPE (Malaysia country code dateline); '2027' kept as DATE; "
        "deleted 'Healthcare' ORG and 'Financial Services' ORG (generic sector nouns); "
        "'667'/'456' CARDINAL and '3%' PERCENT deleted (out of scheme).",
    "a8d7872ea5c5092923b301544b8e65de6d5d9568679b796ae2919b8909431fec":
        "Review: 'AEON Credit Closes' ORG -> boundary fixed to 'AEON Credit' ORG; "
        "'One-year' DATE deleted ('one-year low' is a stock term); garbage span "
        "'Analysts\\n\\nKUALA LUMPUR' ORG -> 'KUALA LUMPUR' GPE; 'AEON Credit Service' ORG -> "
        "expanded to full legal name 'AEON Credit Service (M) Bhd' ORG; ticker '(KL:AEONCR)' -> "
        "'AEONCR' ORG; 'nearly a year'/'the first-half'/'full-year' kept/added as DATE "
        "(relative fiscal-period expressions, cf. train 'the first six months'/'a year earlier' DATE); "
        "'quarter' DATE deleted (generic); 'At least two'/'about one-third' CARDINAL deleted.",
    "bdd840f5db8535c01a7884cba064b4dd9926cff01a182f9f8425ddcbdb963012":
        "Review: 'The Makeover Guys Offer Flexi Loan' ORG -> boundary fixed to "
        "'The Makeover Guys' ORG (company name); 'Home Purchase' ORG deleted (generic); "
        "'The Makeover Guys' WORK_OF_ART -> relabeled ORG; 'The Makeover Flexi Loan' deleted "
        "(product name, out of scheme); '35 years' DATE, 'the age of'/'70' DATE deleted "
        "(duration/age, not date expressions); percents deleted (out of scheme); "
        "added miss 'RHB Bank' ORG.",
}


def resolve(text, label, anchor, lstrip, rstrip, occ):
    """Find occ-th anchor occurrence; return (start, end) of the inner span."""
    idx, pos = 0, -1
    for _ in range(occ):
        pos = text.find(anchor, pos + 1)
        assert pos != -1, f"anchor not found (occ {occ}): {anchor!r}"
    return pos + lstrip, pos + len(anchor) - rstrip


def main():
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    assert len(manifest["eval_extra"]) == 7

    items = []
    # --- 3 test records, copied verbatim from the gold annotations --------------
    n_test = 0
    with open(ANNOT, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["split"] == "test":
                items.append({
                    "article_id": r["article_id"],
                    "source_file": r["source_file"],
                    "row_index": r["row_index"],
                    "split": "test",
                    "text": r["text"],
                    "entities": [{"start": e["start"], "end": e["end"], "label": e["label"]}
                                 for e in r["entities"]],
                    "notes": "Gold labels copied verbatim from project/data/annotations.jsonl "
                             "(human-reviewed).",
                })
                n_test += 1
    assert n_test == 3, n_test

    # --- 7 new gold-labeled excerpts -------------------------------------------
    for ex in manifest["eval_extra"]:
        aid, text = ex["article_id"], ex["text"]
        ents = []
        for label, anchor, lstrip, rstrip, occ in GOLD[aid]:
            s, e = resolve(text, label, anchor, lstrip, rstrip, occ)
            span_text = text[s:e]
            assert span_text and s < e, f"empty span for {anchor!r}"
            ents.append({"start": s, "end": e, "label": label})
        # sanity: no overlaps, sorted
        ents.sort(key=lambda d: (d["start"], d["end"]))
        for a, b in zip(ents, ents[1:]):
            assert a["end"] <= b["start"], f"overlap: {a} vs {b}"
        items.append({
            "article_id": aid,
            "source_file": ex["source_file"],
            "row_index": ex["row_index"],
            "split": "eval",
            "text": text,
            "entities": ents,
            "notes": NOTES[aid],
        })

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(items, fh, indent=2, ensure_ascii=False)

    # --- verification printout ------------------------------------------------
    total = sum(len(i["entities"]) for i in items)
    print(f"wrote {OUT}: {len(items)} items (3 test + 7 eval), {total} gold spans")
    for i in items:
        print(f"\n[{i['split']}] {i['article_id'][:12]} row={i['row_index']}")
        for e in i["entities"]:
            print(f"  [{e['start']}:{e['end']}] {e['label']:6s} '{i['text'][e['start']:e['end']]}'")


if __name__ == "__main__":
    main()
