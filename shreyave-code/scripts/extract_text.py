"""
extract_text.py — Extract text from corpus PDFs using strings command.
Saves to cache/papers_text/P{i}.txt. No API keys required.
"""
import subprocess
import os
import sys

CORPUS_DIR = os.path.join(os.path.dirname(__file__), '..', 'corpus')
CACHE_DIR  = os.path.join(os.path.dirname(__file__), '..', 'cache', 'papers_text')

def extract(pdf_path: str, out_path: str) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result = subprocess.run(['strings', pdf_path], capture_output=True, text=True, errors='replace')
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return len(lines)

def main():
    for i in range(1, 11):
        pdf = os.path.join(CORPUS_DIR, f'P{i}.pdf')
        if not os.path.exists(pdf):
            pdf = os.path.join(CORPUS_DIR, f'P{i}_compressed.pdf')
        if not os.path.exists(pdf):
            print(f'  P{i}: PDF not found in {CORPUS_DIR}', file=sys.stderr)
            continue
        out = os.path.join(CACHE_DIR, f'P{i}.txt')
        n = extract(pdf, out)
        print(f'  P{i}: {n} lines → {out}')

if __name__ == '__main__':
    main()
