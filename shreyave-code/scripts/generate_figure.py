"""
generate_figure.py - Regenerate taxonomy tree diagram without numpy/matplotlib.
Output: taxonomy_figure.png in the project root.
"""
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'taxonomy_figure.png')


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _box(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, text: str, fill: str, txt: str):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill=fill, outline="#1E3A5F", width=2)
    draw.multiline_text((x + 12, y + 10), text, fill=txt, font=_font(18), spacing=4)


def _arrow(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int):
    draw.line((x1, y1, x2, y2), fill="#2C5F8A", width=3)
    draw.polygon([(x2, y2), (x2 - 8, y2 - 5), (x2 - 8, y2 + 5)], fill="#2C5F8A")


def main():
    img = Image.new("RGB", (2200, 1200), "#F8FAFE")
    draw = ImageDraw.Draw(img)

    _box(draw, 580, 40, 1040, 90, "ROOT: LLM Agents for Scientific Research", "#1E3A5F", "white")

    l1 = [
        (110, 240, 560, 110, "A. Knowledge Encoding Approaches"),
        (810, 240, 580, 110, "B. Agentic Reasoning and Tool Use"),
        (1540, 240, 520, 110, "C. Evaluation Frameworks"),
    ]
    for x, y, w, h, t in l1:
        _box(draw, x, y, w, h, t, "#2C5F8A", "white")
        _arrow(draw, 1100, 130, x + w // 2, y)

    leaf = [
        (60, 470, "A.1 Parametric Scientific Pretraining\nGalactica [P9], BioGPT [P10]"),
        (360, 470, "A.2 Symbolic Knowledge Graphs\nSciAgents [P2]"),
        (700, 470, "B.1 Self-Supervised Tool Learning\nToolformer [P5]"),
        (990, 470, "B.2 Reasoning-Action Frameworks\nReAct [P6]"),
        (1280, 470, "B.3 Verbal Reinforcement Learning\nReflexion [P7]"),
        (1570, 470, "B.4 Domain Tool Integration\nChemCrow [P4], PaperQA [P3]"),
        (1680, 720, "C.1 End-to-End Pipeline Eval\nAI Scientist [P1]"),
        (1900, 720, "C.2 Benchmarked Research Tasks\nMLGym [P8]"),
    ]
    for x, y, text in leaf:
        _box(draw, x, y, 260, 170, text, "#CFE8F5", "#1E3A5F")

    for x, y, _ in leaf[:2]:
        _arrow(draw, 390, 350, x + 130, y)
    for x, y, _ in leaf[2:6]:
        _arrow(draw, 1100, 350, x + 130, y)
    for x, y, _ in leaf[6:]:
        _arrow(draw, 1800, 350, x + 130, y)

    draw.text(
        (420, 1120),
        "Figure 1. Taxonomy of LLM Agents for Scientific Research (2-level hierarchy, 10/10 papers cited)",
        fill="#444444",
        font=_font(20),
    )
    img.save(OUT, "PNG")
    print(f"Figure saved -> {OUT} ({os.path.getsize(OUT)//1024} KB)")


if __name__ == "__main__":
    main()
