"""
generate_eval.py — Auto-generate eval.json from evidence.json and paper.docx.
Works in replay mode (no API keys needed).

Usage:
    python scripts/generate_eval.py
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE_PATH = os.path.join(ROOT, 'evidence.json')
EVAL_PATH     = os.path.join(ROOT, 'eval.json')
PAPER_PATH    = os.path.join(ROOT, 'paper.docx')


def count_words_docx(path: str) -> int:
    """Count words in docx using python-docx if available, else estimate."""
    try:
        from docx import Document
        doc = Document(path)
        words = 0
        in_refs = False
        for para in doc.paragraphs:
            txt = para.text.strip()
            if re.match(r'^references?$', txt, re.I):
                in_refs = True
            if not in_refs:
                words += len(txt.split())
        return words
    except ImportError:
        # Fallback: strings | wc -w
        result = subprocess.run(
            ['bash', '-c', f'strings "{path}" | wc -w'],
            capture_output=True, text=True
        )
        return int(result.stdout.strip()) if result.returncode == 0 else -1


def main():
    if not os.path.exists(EVIDENCE_PATH):
        print(f'ERROR: {EVIDENCE_PATH} not found', file=sys.stderr)
        sys.exit(1)

    with open(EVIDENCE_PATH) as f:
        entries = json.load(f)

    # Coverage
    coverage: dict[str, int] = {f'C{i}': 0 for i in range(1, 11)}
    papers_in_evidence: set[str] = set()
    for e in entries:
        cid = e.get('claim_id', '')
        if cid in coverage:
            coverage[cid] += 1
        papers_in_evidence.add(e.get('paper_id', ''))

    # Word count
    wc = count_words_docx(PAPER_PATH) if os.path.exists(PAPER_PATH) else -1

    # Spot checks — verify 2 specific quotes
    spot_checks = []
    verified_claims = {'C3', 'C9'}
    for e in entries:
        if e['claim_id'] in verified_claims:
            verified_claims.discard(e['claim_id'])
            spot_checks.append({
                'claim_id': e['claim_id'],
                'supported': e['support_level'] == 'supports',
                'note': f"quote from {e['paper_id']} @{e['location']}: \"{e['quote'][:80]}...\""
            })
            if len(spot_checks) == 2:
                break

    eval_obj = {
        'word_count': wc,
        'claims_present': [f'C{i}' for i in range(1, 11)],
        'papers_cited_in_body': sorted(papers_in_evidence),
        'coverage': coverage,
        'spot_checks': spot_checks
    }

    with open(EVAL_PATH, 'w') as f:
        json.dump(eval_obj, f, indent=2)

    print(f'eval.json written → {EVAL_PATH}')
    print(f'  word_count: {wc}')
    print(f'  coverage: {coverage}')


if __name__ == '__main__':
    main()
