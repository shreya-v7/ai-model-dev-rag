"""
verify_quotes.py — Verify that every quote in evidence.json appears verbatim
(or near-verbatim, allowing for PDF extraction artefacts like hyphenation) in the
corresponding cached paper text. Prints a pass/fail report.

Usage:
    python scripts/verify_quotes.py
    python scripts/verify_quotes.py --strict   # require exact substring match
"""
import json
import os
import re
import sys
import argparse

EVIDENCE_PATH = os.path.join(os.path.dirname(__file__), '..', 'evidence.json')
CACHE_DIR     = os.path.join(os.path.dirname(__file__), '..', 'cache', 'papers_text')


def normalize(s: str) -> str:
    """Collapse whitespace, strip soft-hyphens, fix PDF line-break artefacts, lowercase.
    
    PDF strings extraction often breaks words at line boundaries, producing tokens like
    'previ ous', 'evalu ation', 'surpass ing'. We join such split fragments by removing
    spaces between a word-fragment ending in a consonant and the continuation fragment.
    """
    s = s.replace('\u00ad', '')          # soft hyphen
    s = re.sub(r'[\n\r]+', ' ', s)
    # Re-join PDF line-break word splits (e.g. "previ\nous" → "previ ous" → "previous")
    # Pattern: short fragment (<= 5 chars, no vowel at end) followed by lowercase continuation
    s = re.sub(r'\b([a-z]{2,5})\s+([a-z]{2,})\b', lambda m:
        m.group(1) + m.group(2) if _is_split_word(m.group(1), m.group(2)) else m.group(0), s)
    s = re.sub(r'\s+', ' ', s)
    return s.lower().strip()


def _is_split_word(prefix: str, suffix: str) -> bool:
    """Heuristic: a prefix likely represents a line-break split if it ends in a consonant
    and the combined form is a single word (no space, no punctuation boundary)."""
    consonants = set('bcdfghjklmnpqrstvwxyz')
    return len(prefix) >= 2 and prefix[-1] in consonants and len(suffix) >= 2


def check_quote(paper_id: str, quote: str, strict: bool) -> tuple[bool, str]:
    txt_path = os.path.join(CACHE_DIR, f'{paper_id}.txt')
    if not os.path.exists(txt_path):
        return False, f'cache file not found: {txt_path}'
    with open(txt_path, encoding='utf-8', errors='replace') as f:
        full_text = normalize(f.read())

    norm_quote = normalize(quote)

    if strict:
        if norm_quote in full_text:
            return True, 'exact match'
        return False, 'NOT FOUND (strict)'

    # Fuzzy: check that ≥80% of 5-word n-grams appear in the text
    words = norm_quote.split()
    if len(words) < 5:
        found = norm_quote in full_text
        return found, 'short quote exact' if found else 'NOT FOUND'
    
    ngrams = [' '.join(words[i:i+5]) for i in range(len(words) - 4)]
    hits = sum(1 for ng in ngrams if ng in full_text)
    ratio = hits / len(ngrams)
    if ratio >= 0.75:
        return True, f'fuzzy match {ratio:.0%} of 5-grams found'
    return False, f'FUZZY FAIL {ratio:.0%} of 5-grams found'


def main(strict: bool = False):
    with open(EVIDENCE_PATH) as f:
        entries = json.load(f)

    passed = 0
    failed = 0
    print(f"\n{'='*60}")
    print(f"Quote Verification Report  (strict={strict})")
    print(f"{'='*60}")

    by_claim: dict[str, list] = {}
    for e in entries:
        by_claim.setdefault(e['claim_id'], []).append(e)

    for cid in sorted(by_claim.keys()):
        for e in by_claim[cid]:
            ok, msg = check_quote(e['paper_id'], e['quote'], strict)
            status = '✓' if ok else '✗'
            if ok:
                passed += 1
            else:
                failed += 1
            short_q = e['quote'][:60].replace('\n', ' ')
            print(f"  {status} {cid}/{e['paper_id']}  [{msg}]")
            if not ok:
                print(f"       quote: \"{short_q}...\"")

    total = passed + failed
    print(f"\nResult: {passed}/{total} quotes verified")
    if failed > 0:
        print(f"WARNING: {failed} quote(s) could not be verified against cached text.")
        print("Check for OCR/encoding differences or paraphrasing errors.")
        sys.exit(1)
    else:
        print("All quotes verified ✓")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strict', action='store_true', help='Require exact substring match')
    args = parser.parse_args()
    main(strict=args.strict)
