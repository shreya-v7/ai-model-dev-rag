# Prompts Log

## Automation and Reproducibility Notes

This submission supports **Replay Mode**. All corpus PDFs were processed locally using
`strings` extraction (no authenticated APIs). The `evidence.json` quotes were extracted
directly from the local PDF text and verified against the raw strings output.
The `paper.docx` is generated via `node scripts/build_paper.js` using the `docx` npm package.

To reproduce all artifacts without any API keys:
```
python scripts/run_all.py --mode replay
```

---

## Prompt 1: Corpus Structure Analysis

**Tool/Model:** Claude (claude-sonnet-4-6), internal reasoning  
**Purpose:** Understand the intellectual landscape of the 10 corpus papers and identify synthesis themes that cut across multiple papers.

**Prompt (paraphrase):**
> "Read the extracted text from P1–P10. Identify three synthesis themes that each span at least 3 papers. Each theme should: (a) open with a claim about what papers collectively show, (b) cite ≥2 papers with specific evidence, (c) close with a tension or limitation. Do NOT write one paragraph per paper."

**Outcome:** Three themes identified — (A) Scientific Domain Pretraining [P9, P10, P2], (B) Tool Use and Iterative Reasoning [P5, P6, P7, P3, P4], (C) Evaluation and the Novelty Gap [P1, P8].

---

## Prompt 2: Verbatim Quote Extraction

**Tool/Model:** PDF strings extraction (bash `strings` command on local PDFs)  
**Purpose:** Extract verbatim quotes for evidence.json, traceable to exact page/section of each corpus PDF.

**Commands used:**
```bash
strings /mnt/project/P6.pdf | grep -n "34\|ALFWorld\|absolute"
strings /mnt/project/P7.pdf | grep -n "91.*pass\|HumanEval"
strings /mnt/project/P8.pdf | grep -n "hyperparameter\|novel.*algorithm"
# (repeated for all 10 papers, targeting claim-specific numerical anchors)
```

**Verification:** Each quote was cross-checked by finding the surrounding paragraph in the strings output to confirm the quote is verbatim and not paraphrased.

---

## Prompt 3: Critical Flaw Identification (Beyond Abstract-Level)

**Tool/Model:** Claude (claude-sonnet-4-6)  
**Purpose:** Identify genuine methodological or logical flaws in each paper's research design, beyond surface-level observations.

**Prompt (paraphrase):**
> "For each of the 10 corpus papers, identify one specific methodological, logical, or evaluation design flaw that would not be visible from reading only the abstract. The flaw must be: (a) specific to that paper's experimental design, (b) different from what the paper itself acknowledges as a limitation, (c) tied to a specific claim or experimental setup. Then embed these flaws in the claims table and future directions."

**Outcomes (key flaws identified):**
- P1: Circular evaluation — reviewer calibrated on same distributional signature as generated outputs.
- P3: Author-designed benchmark (LitQA) structurally advantages retrieval agents.
- P4: Evaluator invalidation used as quality proof — logically inverted.
- P6: Model-scale confound in RL baseline comparison.
- P7: Self-generated unit tests as success criterion may produce false positives.
- P8: Six-level capability hierarchy has only Level 1 empirically populated.
- P9: Galactica retracted for misinformation; benchmarks did not test for this failure mode.

---

## Prompt 4: Taxonomy Construction

**Tool/Model:** Claude (claude-sonnet-4-6) + matplotlib (local, no API)  
**Purpose:** Build a 2-level taxonomy with 10+ nodes, 7+ corpus citations, per-node definitions, and a matching diagram.

**Prompt (paraphrase):**
> "Organise all 10 corpus papers into a 2-level taxonomy rooted at 'LLM Agents for Scientific Research'. Requirements: (1) every node has parent except root; (2) ≥10 total nodes; (3) each leaf node has a 1–2 sentence definition + [P#] citation; (4) cite ≥7 papers across taxonomy; (5) sibling nodes must be meaningfully distinct. Generate both the text taxonomy and a matplotlib tree diagram embedded in the Word document."

**Figure generation command:**
```bash
python3 scripts/generate_figure.py  # saves taxonomy_figure.png at 200 DPI
```

---

## Prompt 5: Future Directions Gap-Grounding Check

**Tool/Model:** Claude (claude-sonnet-4-6), self-evaluation against rubric  
**Purpose:** Ensure each direction's Gap statement cites a specific corpus paper that explicitly identifies or implies the limitation.

**Prompt (paraphrase):**
> "For each of the 5 future directions, verify: (a) the Gap statement names a specific corpus paper and a specific finding from that paper; (b) the Approach is concrete (names a technique or system, not just a category); (c) the Evaluation names a specific benchmark, metric, or experimental design. Reject directions that are speculative research proposals not directly grounded in a corpus finding."

**Tightened directions as a result:** Direction 2 (tool selection) was revised to cite Toolformer's perplexity filter as the explicit mechanism creating the gap. Direction 5 (KG fusion) was revised to cite P2's explicit limitation of being single-domain.
