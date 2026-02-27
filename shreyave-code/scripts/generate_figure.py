"""
generate_figure.py — Regenerate the taxonomy tree diagram.
Output: taxonomy_figure.png (200 DPI) in the project root.
No API keys required.

Usage:
    python scripts/generate_figure.py
"""
import os
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
except ImportError:
    print('ERROR: matplotlib not installed. Run: pip install matplotlib', file=sys.stderr)
    sys.exit(1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, 'taxonomy_figure.png')


def draw_box(ax, cx, cy, w, h, text, fc, tc='white', fs=9, bold=True):
    r = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle="round,pad=0.12",
                       facecolor=fc, edgecolor='#1E3A5F', linewidth=1.8, zorder=3)
    ax.add_patch(r)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            color=tc, fontweight='bold' if bold else 'normal',
            multialignment='center', linespacing=1.35, zorder=4)


def draw_edge(ax, px, py, ph, cx, cy, ch):
    ax.annotate('', xy=(cx, cy+ch/2), xytext=(px, py-ph/2),
                arrowprops=dict(arrowstyle='->', color='#2C5F8A', lw=1.7), zorder=2)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(18, 8.5))
    ax.set_xlim(0, 18); ax.set_ylim(0, 8.5); ax.axis('off')
    ax.set_facecolor('#F8FAFE'); fig.patch.set_facecolor('#F8FAFE')

    # Root
    draw_box(ax, 9.0, 7.9, 9.5, 0.75,
             'ROOT: LLM Agents for Scientific Research',
             '#1E3A5F', 'white', fs=11.5)

    # Level 1
    L1 = [
        ( 2.5, 6.15, 3.8, 0.9, 'A. Knowledge\nEncoding Approaches',   '#2C5F8A'),
        ( 9.0, 6.15, 4.2, 0.9, 'B. Agentic Reasoning & Tool Use',     '#2C5F8A'),
        (15.4, 6.15, 3.4, 0.9, 'C. Evaluation Frameworks',             '#2C5F8A'),
    ]
    for cx, cy, w, h, lbl, col in L1:
        draw_box(ax, cx, cy, w, h, lbl, col, 'white', fs=9.5)
        draw_edge(ax, 9.0, 7.9, 0.75, cx, cy, h)

    # Level 2 — A
    A2 = [
        (1.2, 4.3, 2.4, 1.65,
         'A.1 Parametric\nScientific Pretraining\n─────────\nGalactica [P9]\nBioGPT [P10]',
         '#CFE8F5', '#1E3A5F'),
        (3.8, 4.3, 2.4, 1.65,
         'A.2 Symbolic\nKnowledge Graphs\n─────────\nSciAgents [P2]',
         '#CFE8F5', '#1E3A5F'),
    ]
    for cx, cy, w, h, lbl, fc, tc in A2:
        draw_box(ax, cx, cy, w, h, lbl, fc, tc, fs=8.3, bold=False)
        draw_edge(ax, 2.5, 6.15, 0.9, cx, cy, h)

    # Level 2 — B
    B2 = [
        ( 5.6, 4.25, 2.25, 1.7,
          'B.1 Self-Supervised\nTool Learning\n─────────\nToolformer [P5]',
          '#CFE8F5', '#1E3A5F'),
        ( 7.9, 4.25, 2.25, 1.7,
          'B.2 Reasoning–Action\nFrameworks\n─────────\nReAct [P6]',
          '#CFE8F5', '#1E3A5F'),
        (10.2, 4.25, 2.25, 1.7,
          'B.3 Verbal\nReinforcement\nLearning\n─────────\nReflexion [P7]',
          '#CFE8F5', '#1E3A5F'),
        (12.5, 4.25, 2.4, 1.7,
          'B.4 Domain-Specific\nTool Integration\n─────────\nChemCrow [P4]\nPaperQA [P3]',
          '#CFE8F5', '#1E3A5F'),
    ]
    for cx, cy, w, h, lbl, fc, tc in B2:
        draw_box(ax, cx, cy, w, h, lbl, fc, tc, fs=8.0, bold=False)
        draw_edge(ax, 9.0, 6.15, 0.9, cx, cy, h)

    # Level 2 — C
    C2 = [
        (14.2, 4.3, 2.3, 1.65,
         'C.1 End-to-End\nPipeline Evaluation\n─────────\nAI Scientist [P1]',
         '#CFE8F5', '#1E3A5F'),
        (16.5, 4.3, 2.3, 1.65,
         'C.2 Benchmarked\nResearch Tasks\n─────────\nMLGym [P8]',
         '#CFE8F5', '#1E3A5F'),
    ]
    for cx, cy, w, h, lbl, fc, tc in C2:
        draw_box(ax, cx, cy, w, h, lbl, fc, tc, fs=8.3, bold=False)
        draw_edge(ax, 15.4, 6.15, 0.9, cx, cy, h)

    # Legend
    patches = [
        mpatches.Patch(color='#1E3A5F', label='Root node'),
        mpatches.Patch(color='#2C5F8A', label='Level 1 — Category'),
        mpatches.Patch(facecolor='#CFE8F5', edgecolor='#1E3A5F', label='Level 2 — System / Method'),
    ]
    ax.legend(handles=patches, loc='lower left', bbox_to_anchor=(0.0, 0.0),
              fontsize=9, framealpha=0.95, edgecolor='#AAAAAA',
              title='Legend', title_fontsize=9.2)

    # Caption
    ax.text(9, 0.3,
            'Figure 1.  Taxonomy of LLM Agents for Scientific Research  '
            '(2-level hierarchy · 13 nodes · 10/10 corpus papers cited)',
            ha='center', va='center', fontsize=9.5, color='#444444', style='italic')

    plt.tight_layout(pad=0.3)
    plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='#F8FAFE')
    plt.close()
    print(f'Figure saved → {OUT}  ({os.path.getsize(OUT)//1024} KB)')


if __name__ == '__main__':
    main()
