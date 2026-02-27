"""
run_all.py - Main entry point for reproducibility and offline RAG pipeline.

Modes:
  --mode replay       Deterministic rebuild from existing artifacts, no API keys
  --mode offline_rag  End-to-end local RAG from cache, no API keys
  --mode full         Extract cache from PDFs, then run replay
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, 'scripts')
OUT_ROOT = os.path.join(ROOT, 'outputs')
CORPUS_DIR = os.path.join(ROOT, 'corpus')


def run(cmd: list, label: str):
    print(f'\n[{label}]')
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f'  ERROR: {label} failed with code {result.returncode}', file=sys.stderr)
        sys.exit(result.returncode)
    print(f'  OK')


def sync_artifacts(space: str):
    target = os.path.join(OUT_ROOT, space)
    os.makedirs(target, exist_ok=True)
    artifacts = [
        'paper.docx',
        'evidence.json',
        'eval.json',
        'prompts.md',
        'README.md',
        'taxonomy_figure.png',
    ]
    copied = 0
    for name in artifacts:
        src = os.path.join(ROOT, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(target, name))
            copied += 1
    print(f'  Copied {copied} artifacts to {target}')


def _corpus_ready() -> bool:
    required = [os.path.join(CORPUS_DIR, f'P{i}.pdf') for i in range(1, 11)]
    return all(os.path.exists(path) for path in required)


def replay_mode():
    """Reproduce all artifacts from cached corpus text — no API keys."""
    print('=== REPLAY MODE (no API keys required) ===')

    # 1. Verify quotes
    run([sys.executable, os.path.join(SCRIPTS, 'verify_quotes.py')],
        'Step 1: Verify evidence.json quotes against cached paper text')

    # 2. Regenerate eval.json
    run([sys.executable, os.path.join(SCRIPTS, 'generate_eval.py')],
        'Step 2: Regenerate eval.json')

    # 3. Regenerate figure
    run([sys.executable, os.path.join(SCRIPTS, 'generate_figure.py')],
        'Step 3: Regenerate taxonomy figure')

    # 4. Rebuild paper.docx
    build_js = os.path.join(SCRIPTS, 'build_paper.js')
    if os.path.exists(build_js):
        run(['node', build_js],
            'Step 4: Rebuild paper.docx')
    else:
        print('  [Step 4] build_paper.js not found — skipping docx rebuild')

    sync_artifacts('offline')
    print('\n=== REPLAY COMPLETE ===')
    print('Artifacts: evidence.json, eval.json, taxonomy_figure.png, paper.docx')
    print(f'Offline artifact space: {os.path.join(OUT_ROOT, "offline")}')


def offline_rag_mode():
    """Run end-to-end local RAG from cached corpus, then rebuild artifacts."""
    print('=== OFFLINE RAG MODE (no API keys required) ===')

    run([sys.executable, os.path.join(SCRIPTS, 'generate_evidence_rag.py')],
        'Step 1: Build evidence.json via local RAG')
    run([sys.executable, os.path.join(SCRIPTS, 'verify_quotes.py')],
        'Step 2: Verify generated evidence quotes')
    run([sys.executable, os.path.join(SCRIPTS, 'generate_eval.py')],
        'Step 3: Regenerate eval.json')
    run([sys.executable, os.path.join(SCRIPTS, 'generate_figure.py')],
        'Step 4: Regenerate taxonomy figure')

    build_js = os.path.join(SCRIPTS, 'build_paper.js')
    if os.path.exists(build_js):
        run(['node', build_js], 'Step 5: Rebuild paper.docx')
    else:
        print('  [Step 5] build_paper.js not found - skipping docx rebuild')

    sync_artifacts('offline')
    print('\n=== OFFLINE RAG COMPLETE ===')
    print(f'Offline artifact space: {os.path.join(OUT_ROOT, "offline")}')


def full_mode():
    """Full pipeline: extract text → retrieve quotes → verify → generate docs."""
    print('=== FULL PIPELINE MODE ===')
    print('Loading environment variables from .env ...')

    if not _corpus_ready():
        print('WARNING: corpus/P1..P10.pdf not found. Falling back to replay mode.', file=sys.stderr)
        replay_mode()
        return

    try:
        from dotenv import load_dotenv
        env_path = os.path.join(ROOT, '.env')
        if not os.path.exists(env_path):
            print('ERROR: .env not found. Copy .env.example to .env and fill in keys.',
                  file=sys.stderr)
            sys.exit(1)
        load_dotenv(env_path)
    except ImportError:
        print('WARNING: python-dotenv not installed. Reading os.environ directly.')

    # Step 1: Extract text
    run([sys.executable, os.path.join(SCRIPTS, 'extract_text.py')],
        'Step 1: Extract PDF text to cache/papers_text/')

    # Steps 2-4: same as replay
    replay_mode()
    sync_artifacts('online')
    print(f'Online artifact space: {os.path.join(OUT_ROOT, "online")}')


def main():
    parser = argparse.ArgumentParser(description='Mini-survey reproducibility pipeline')
    parser.add_argument(
        '--mode',
        choices=['replay', 'offline_rag', 'full'],
        default='replay',
        help='replay = deterministic rebuild; offline_rag = local end-to-end RAG; full = refresh cache from PDFs then replay',
    )
    args = parser.parse_args()

    if args.mode == 'full':
        full_mode()
    elif args.mode == 'offline_rag':
        offline_rag_mode()
    else:
        replay_mode()


if __name__ == '__main__':
    main()
