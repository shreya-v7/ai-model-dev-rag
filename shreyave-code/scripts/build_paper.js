const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  VerticalAlign, ImageRun, UnderlineType
} = require('docx');
const fs = require('fs');
const path = require('path');

// ── helpers ──────────────────────────────────────────────────────────────────
const bdr   = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const bdrs  = { top: bdr, bottom: bdr, left: bdr, right: bdr };
const hBdr  = { style: BorderStyle.SINGLE, size: 5, color: "1E3A5F" };
const hBdrs = { top: hBdr, bottom: hBdr, left: bdr, right: bdr };

function run(text, opts = {}) {
  return new TextRun({ text, size: 22, font: "Arial", ...opts });
}
function bold(text) { return run(text, { bold: true }); }
function italic(text) { return run(text, { italics: true }); }

function p(...runs) {
  return new Paragraph({
    children: Array.isArray(runs[0]) ? runs[0] : runs,
    spacing: { before: 100, after: 100 },
    alignment: AlignmentType.JUSTIFIED
  });
}
function plain(text) { return p(run(text)); }

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, bold: true, size: 30, font: "Arial", color: "1E3A5F" })],
    spacing: { before: 360, after: 180 }
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, bold: true, size: 24, font: "Arial", color: "2C5F8A" })],
    spacing: { before: 240, after: 100 }
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text, bold: true, size: 22, font: "Arial", color: "2C5F8A" })],
    spacing: { before: 180, after: 80 }
  });
}
function sp() {
  return new Paragraph({ children: [], spacing: { before: 60, after: 60 } });
}

// Table cells
function hCell(text, w) {
  return new TableCell({
    borders: hBdrs,
    width: { size: w, type: WidthType.DXA },
    shading: { fill: "1E3A5F", type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      children: [new TextRun({ text, bold: true, color: "FFFFFF", size: 19, font: "Arial" })],
      alignment: AlignmentType.CENTER
    })]
  });
}
function dCell(paras, w, shade = false) {
  const children = typeof paras === 'string'
    ? [new Paragraph({ children: [run(paras)], spacing: { before: 40, after: 40 } })]
    : paras;
  return new TableCell({
    borders: bdrs,
    width: { size: w, type: WidthType.DXA },
    shading: { fill: shade ? "EBF0F7" : "FFFFFF", type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children
  });
}
function cp(...runs) {
  return new Paragraph({
    children: runs,
    spacing: { before: 30, after: 30 }
  });
}
function tr(cells) { return new TableRow({ children: cells }); }

// ── image ────────────────────────────────────────────────────────────────────
const FIG_PATH = path.join(__dirname, '..', 'taxonomy_figure.png');
const figData = fs.readFileSync(FIG_PATH);

// ── DOCUMENT ─────────────────────────────────────────────────────────────────
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } }
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1260, bottom: 1440, left: 1260 }
      }
    },
    children: [

      // ───────────────────────────────────────────────────────────────────────
      // TITLE
      // ───────────────────────────────────────────────────────────────────────
      new Paragraph({
        children: [new TextRun({
          text: "From Tools to Discoveries: A Mini-Survey of LLM Agents for Scientific Research",
          bold: true, size: 36, font: "Arial", color: "1E3A5F"
        })],
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 160 }
      }),
      new Paragraph({
        children: [new TextRun({
          text: "Mini-Survey · Corpus: P1–P10 · Comparison: S2",
          size: 22, font: "Arial", color: "666666", italics: true
        })],
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 520 }
      }),

      // ═══════════════════════════════════════════════════════════════════════
      // SECTION 1 — LITERATURE SUMMARY  (~780 words)
      // ═══════════════════════════════════════════════════════════════════════
      h1("1. Literature Summary"),

      h2("Theme 1: Scientific Domain Pretraining and Knowledge Encoding"),

      p(run("A recurring finding across the corpus is that general-purpose LLMs perform inadequately on technical scientific tasks, and that dedicated domain pretraining substantially closes that gap—though at costs that deserve scrutiny. Galactica "), bold("[P9]"), run(" demonstrates this most directly: trained on over 48 million curated scientific documents with domain-specific tokenization for SMILES strings, LaTeX equations, and citation markers, its 120B model scores 20.4% on the MATH benchmark versus PaLM 540B's 8.8%, while the 30B variant outperforms PaLM 540B with 18 times fewer parameters. A critical gap in this evaluation is that the benchmark suite contains no test for open-ended generation quality or misinformation detection—a mismatch between curated benchmarks and deployment reliability that the paper's own metrics cannot surface. BioGPT "), bold("[P10]"), run(" reinforces the specialization argument at smaller scale: a 347M-parameter model pre-trained from scratch on 15 million PubMed abstracts achieves 44.98% F1 on BC5CDR relation extraction and 78.2% on PubMedQA. A substantive limitation here is that training on abstracts but evaluating on full-paper tasks means performance gains may partially reflect vocabulary alignment rather than genuine biomedical reasoning. SciAgents "), bold("[P2]"), run(" takes a complementary symbolic route, encoding domain knowledge as an ontological knowledge graph derived from approximately 1,000 biology papers, using hierarchical subgraph sampling to ground multi-agent hypothesis generation in mechanistically explicit context. However, novelty and feasibility assessment is tool-mediated and author-directed, with no clearly independent blind external panel or operationalized quantitative novelty criterion. Together, these three papers establish that encoding domain knowledge—whether parametrically or symbolically—yields measurable gains, but the persistent disconnect between benchmark accuracy and deployed reliability remains the field's deepest unresolved tension.")),

      h2("Theme 2: Tool Use, Retrieval, and Iterative Reasoning Frameworks"),

      p(run("A second theme addresses how LLMs are extended beyond static knowledge via tool integration, retrieval augmentation, and iterative self-correction. Toolformer "), bold("[P5]"), run(" establishes a principled self-supervised baseline: a perplexity-reduction filter selects API calls (calculator, QA, Wikipedia, machine translation, calendar) that genuinely help predict future tokens, enabling a 6.7B GPT-J model to match or outperform GPT-3 (175B) on zero-shot math and LAMA benchmarks. A structural flaw the paper does not address is that this filter preferentially selects tools for information already present in training text, precisely the opposite of what scientific discovery demands—tools are most urgently needed for knowledge absent from training data (new experiments, live databases). ReAct "), bold("[P6]"), run(" extends tool use by interleaving language reasoning traces with environmental actions, achieving 34% absolute improvement over RL baselines on ALFWorld; however, this comparison is confounded by model scale—ReAct uses PaLM-540B while the baselines were trained from scratch. Reflexion "), bold("[P7]"), run(" achieves 91% pass@1 on HumanEval through verbal self-reflection in episodic memory, but uses self-generated unit tests to determine trial success: an agent can write tests its own faulty code passes while still failing on edge cases. PaperQA "), bold("[P3]"), run(" beats GPT-4 by 30 percentage points on closed-book PubMedQA (86.3% vs. 57.9%) via modular agentic RAG, though its primary novelty benchmark (LitQA) was designed by its own authors, creating a construct validity concern. ChemCrow "), bold("[P4]"), run(" integrates 13 expert-designed chemistry tools with GPT-4 for complex synthesis tasks; notably, the paper presents GPT-4's inability to distinguish correct ChemCrow outputs from wrong GPT-4 completions as evidence of ChemCrow's quality—when it is logically evidence of the evaluator's inadequacy. Collectively, these frameworks reveal a shared epistemological problem: tool-augmented agents are validated either by the models that built them or on benchmarks designed by their creators.")),

      h2("Theme 3: Evaluating Scientific AI Agents and the Novelty Gap"),

      p(run("The third theme addresses what it means to evaluate scientific AI capability and what those evaluations actually reveal. The AI Scientist "), bold("[P1]"), run(" presents the most ambitious target: a fully automated pipeline producing complete ML research papers at under $15 each, with a GPT-4o-based reviewer achieving 65% balanced accuracy versus 66% for human reviewers on ICLR 2022 papers. A structural flaw in this evaluation is circularity—the reviewer is calibrated on ICLR 2022 submissions, and The AI Scientist generates papers explicitly structured to resemble ICLR submissions. Both share the same distributional signature, inflating apparent near-human performance without testing whether either system detects genuine scientific insight. MLGym "), bold("[P8]"), run(" provides a more rigorous perspective: frontier models—Claude-3.5-Sonnet, GPT-4o, o1-preview, Llama-3.1 405B, Gemini-1.5 Pro—evaluated on 13 open-ended ML research tasks improve on baselines only through hyperparameter search, never generating novel algorithms or architectures. The paper proposes a six-level capability hierarchy (Level 0: reproduction → Level 5: paradigm-shifting research), but this hierarchy is theoretical: only Level 1 is empirically populated, and the upper levels lack any falsifiable criteria distinguishing them. Together, P1 and P8 establish a fundamental novelty gap: current LLM agents can optimize within a defined solution space but cannot redefine it. Neither paper proposes a methodology for detecting whether high benchmark scores reflect genuine scientific reasoning or sophisticated interpolation—the open problem that neither end-to-end pipeline evaluation nor structured benchmarking has yet solved.")),

      sp(),

      // ═══════════════════════════════════════════════════════════════════════
      // SECTION 2 — KEY CLAIMS TABLE  (28 pts)
      // ═══════════════════════════════════════════════════════════════════════
      h1("2. Key Claims Table"),

      new Table({
        width: { size: 9720, type: WidthType.DXA },
        columnWidths: [640, 8000, 1080],
        rows: [
          tr([hCell("ID", 640), hCell("Claim (specific & falsifiable) + Supporting Verbatim Evidence", 8000), hCell("Source", 1080)]),

          tr([
            dCell("C1", 640, true),
            dCell([
              cp(run("ReAct outperforms imitation and reinforcement learning methods on ALFWorld and WebShop by absolute success-rate margins of 34% and 10% respectively, using only 1–2 in-context examples, by interleaving language reasoning traces with environmental actions in an augmented action space. On HotpotQA and Fever, ReAct reduces hallucination and error propagation in chain-of-thought by interacting with a Wikipedia API.")),
              cp(italic('Evidence: "on two interactive decision making benchmarks (ALFWorld and WebShop), ReAct outperforms imitation and reinforcement learning methods by an absolute success rate of 34% and 10% respectively, while being prompted with only one or two in-context examples." — Abstract, P6'))
            ], 8000, true),
            dCell("[P6]", 1080, true)
          ]),

          tr([
            dCell("C2", 640),
            dCell([
              cp(run("Reflexion achieves 91% pass@1 on HumanEval, surpassing GPT-4's 80%, by converting environmental feedback into verbal self-reflections stored in a bounded episodic memory buffer and replayed as additional context in subsequent episodes—without any gradient-based weight updates.")),
              cp(italic('Evidence: "Reflexion achieves a 91% pass@1 accuracy on the HumanEval coding benchmark, surpassing the previous state-of-the-art GPT-4 that achieves 80%." — Abstract, P7'))
            ], 8000),
            dCell("[P7]", 1080)
          ]),

          tr([
            dCell("C3", 640, true),
            dCell([
              cp(run("Toolformer, a 6.7B-parameter GPT-J model fine-tuned via self-supervised perplexity-reduction filtering of sampled API calls, scores 29.4% on SVAMP vs. GPT-3 (175B)'s 10.0%, outperforming a model 25× larger on zero-shot math without any human annotation of tool use.")),
              cp(italic('Evidence: "Model ASDiv SVAMP MAWPS / Toolformer 40.4 29.4 44.0 / GPT-3 (175B) 14.0 10.0 19.8" — Table 4, P5'))
            ], 8000, true),
            dCell("[P5]", 1080, true)
          ]),

          tr([
            dCell("C4", 640),
            dCell([
              cp(run("Frontier LLMs evaluated on MLGym-Bench's 13 open-ended ML research tasks—including Claude-3.5-Sonnet, GPT-4o, o1-preview, Llama-3.1 405B, and Gemini-1.5 Pro—improve over baselines only through hyperparameter search; none generate novel hypotheses, algorithms, or architectures, placing all models at Level 1 of the proposed six-level capability hierarchy.")),
              cp(italic('Evidence: "We find that current frontier models can improve on the given baselines, usually by finding better hyperparameters, but do not generate novel hypotheses, algorithms, architectures, or substantial improvements." — Abstract, P8'))
            ], 8000),
            dCell("[P8]", 1080)
          ]),

          tr([
            dCell("C5", 640, true),
            dCell([
              cp(run("The AI Scientist generates complete ML research papers at under $15 each; its GPT-4o automated reviewer achieves 65% balanced accuracy versus 66% for human reviewers on ICLR 2022 OpenReview data. This near-human performance is partly artefactual: the reviewer was calibrated on the same ICLR distributional signature the system generates into.")),
              cp(italic('Evidence: "Each idea is implemented and developed into a full paper at a meager cost of less than $15 per paper." and "65% vs. 66% balanced accuracy when evaluated on ICLR 2022 OpenReview data." — Abstract + Sec. 4, P1'))
            ], 8000, true),
            dCell("[P1]", 1080, true)
          ]),

          tr([
            dCell("C6", 640),
            dCell([
              cp(run("Galactica (120B), pre-trained on 48M+ scientific documents with SMILES/LaTeX/citation tokens, scores 20.4% on MATH vs. PaLM 540B's 8.8%; the 30B variant outperforms PaLM 540B with 18× fewer parameters. Despite these benchmark results, the paper's evaluation suite contains no direct test for open-ended generation quality or misinformation detection, leaving a reliability gap between benchmark success and deployment behavior.")),
              cp(italic('Evidence: "Our 120B model achieves a score of 20.4% versus PaLM 540B\'s 8.8% on MATH. The 30B model also beats PaLM 540B on this task with 18 times less parameters." — Sec. 1, P9'))
            ], 8000),
            dCell("[P9]", 1080)
          ]),

          tr([
            dCell("C7", 640, true),
            dCell([
              cp(run("BioGPT (347M), a GPT-2 backbone pre-trained from scratch on 15M PubMed abstracts with a 42,384-token in-domain BPE vocabulary, achieves 44.98% F1 on BC5CDR, 38.42% F1 on KD-DTI, and 78.2% accuracy on PubMedQA—establishing SOTA at publication across five biomedical NLP tasks.")),
              cp(italic('Evidence: "we get 44.98%, 38.42% and 40.76% F1 score on BC5CDR, KD-DTI and DDI end-to-end relation extraction tasks respectively, and 78.2% accuracy on PubMedQA, creating a new record." — Abstract, P10'))
            ], 8000, true),
            dCell("[P10]", 1080, true)
          ]),

          tr([
            dCell("C8", 640),
            dCell([
              cp(run("PaperQA, a modular agentic RAG system, surpasses GPT-4 by 30 points on closed-book PubMedQA (86.3% vs. 57.9%) and claims human parity on LitQA; however, LitQA was designed and curated by the PaperQA authors themselves, creating a construct validity concern about the human-parity comparison.")),
              cp(italic('Evidence: "We modified PubMedQA to remove the provided context (so it is closed-book) and found PaperQA beats GPT-4 by 30 points (57.9% to 86.3%)." — Sec. 3, P3'))
            ], 8000),
            dCell("[P3]", 1080)
          ]),

          tr([
            dCell("C9", 640, true),
            dCell([
              cp(run("ChemCrow integrates 13 expert-designed chemistry tools (retrosynthesis, reaction prediction, safety screening, SMILES operations) with GPT-4, enabling multi-step synthesis tasks. Critically, GPT-4 used as evaluator cannot distinguish correct ChemCrow outputs from clearly wrong GPT-4 completions, yet the paper presents this evaluator failure as evidence of ChemCrow's quality rather than as evidence of the evaluator's inadequacy.")),
              cp(italic('Evidence: "we find that GPT-4 as an evaluator cannot distinguish between clearly wrong GPT-4 completions and GPT-4 + ChemCrow performance." — Abstract, P4'))
            ], 8000, true),
            dCell("[P4]", 1080, true)
          ]),

          tr([
            dCell("C10", 640),
            dCell([
              cp(run("SciAgents generates research hypotheses via a multi-agent system grounded in an ontological knowledge graph developed from ~1,000 biology papers, using a novel hierarchical sub-graph sampling strategy; the system produces hypotheses rated for novelty and feasibility through author-directed tooling, but independent blind external expert validation and a quantitative novelty metric (e.g., embedding distance from training corpus) are not clearly reported, making this falsifiable primarily as a process claim rather than an outcome claim.")),
              cp(italic('Evidence: "Central to our hypothesis generation is the utilization of a large ontological knowledge graph, focusing on biological materials, and developed from around 1,000 scientific papers in this domain. We implemented a novel sampling strategy to extract relevant sub-graphs from this comprehensive knowledge graph." — Sec. 1, P2'))
            ], 8000),
            dCell("[P2]", 1080)
          ]),
        ]
      }),

      sp(),

      // ═══════════════════════════════════════════════════════════════════════
      // SECTION 3 — FUTURE DIRECTIONS
      // ═══════════════════════════════════════════════════════════════════════
      h1("3. Directions for Future Research"),

      h3("Direction 1: Adversarial Novelty Testing"),
      p(bold("Gap: "), run("MLGym "), bold("[P8]"), run(" shows all frontier models sit at Level 1; The AI Scientist "), bold("[P1]"), run(" reviewer is calibrated on the same distributional signature it evaluates. Neither can detect training-data interpolation masquerading as discovery.")),
      p(bold("Approach: "), run("Design adversarial probes: inject papers whose conclusions replicate training-data results but with structural surface changes (notation rotation, reversed framing). Measure false-novelty rate—how often the system incorrectly labels interpolations as original contributions.")),
      p(bold("Evaluation: "), run("True/false novelty rates against blind expert panels; literature embedding distance as quantitative novelty proxy; benchmarked against MLGym "), bold("[P8]"), run(" Level 2 threshold.")),

      h3("Direction 2: Knowledge-Gap-Aware Tool Invocation"),
      p(bold("Gap: "), run("Toolformer's "), bold("[P5]"), run(" perplexity filter prefers tools for information already in training text. Scientific discovery requires tools precisely for knowledge the model lacks—new experiments, live literature—which the filter systematically underselects.")),
      p(bold("Approach: "), run("Augment the perplexity filter with a calibrated uncertainty estimator: invoke tools when confidence is low AND the token type matches a scientific-claim pattern (equations, citations, experimental data). Reward tool use proportional to knowledge-gap severity.")),
      p(bold("Evaluation: "), run("Time-stratified benchmarks where training data predates evaluation questions by 12+ months; measure tool-invocation recall, precision, and downstream accuracy relative to Toolformer "), bold("[P5]"), run(" and PaperQA "), bold("[P3]"), run(" baselines.")),

      h3("Direction 3: Third-Party Scientific QA Benchmark Governance"),
      p(bold("Gap: "), run("PaperQA's "), bold("[P3]"), run(" primary novelty benchmark LitQA was designed by its own authors; ChemCrow "), bold("[P4]"), run(" evaluates on 12 hand-selected use cases where its tools were pre-tuned. Both benchmarks structurally favour their creators.")),
      p(bold("Approach: "), run("Pre-registered, third-party benchmark governance: questions submitted by domain experts unaffiliated with any evaluated system, held in escrow, verified by independent reviewers before release—analogous to ICLR OpenReview for model evaluation.")),
      p(bold("Evaluation: "), run("Re-test PaperQA "), bold("[P3]"), run(" and ChemCrow "), bold("[P4]"), run(" on blind third-party questions; compare performance with and without author-designed benchmarks to quantify the benchmark-design advantage empirically.")),

      h3("Direction 4: Long-Horizon Persistent Memory for Multi-Session Research"),
      p(bold("Gap: "), run("Reflexion "), bold("[P7]"), run(" bounds episodic memory to 1–3 episodes due to context limits. MLGym "), bold("[P8]"), run(" shows frontier agents lose optimal configurations over long trajectories, preventing multi-session progress and capping capability at Level 1.")),
      p(bold("Approach: "), run("Hierarchical episodic memory: (a) compressed key-value embedding stores for precise configuration retrieval across sessions; (b) narrative research-diary summaries for strategy planning. Memory writes triggered only by validated performance improvements.")),
      p(bold("Evaluation: "), run("MLGym-Bench "), bold("[P8]"), run(" with extended 48–96h budgets; track improvement curves vs context-bounded Reflexion agents; specifically measure whether Level 2 (SOTA achievement) becomes reachable with persistent memory.")),

      h3("Direction 5: Cross-Domain Knowledge Graph Fusion"),
      p(bold("Gap: "), run("SciAgents "), bold("[P2]"), run(" builds its ontological graph from a single domain (~1,000 biologically-inspired materials papers), preventing cross-disciplinary hypothesis generation. Galactica "), bold("[P9]"), run(" stores cross-domain knowledge parametrically but cannot trace explicit reasoning chains connecting disparate concepts.")),
      p(bold("Approach: "), run("Fuse Galactica-style pretraining (broad concept coverage) with SciAgents-style graphs (explicit relational provenance), constructing multi-domain graphs where every generated hypothesis is traceable to specific graph paths bridging, for example, materials science and drug discovery.")),
      p(bold("Evaluation: "), run("Recruit experts across three paired disciplines to rate hypothesis novelty, feasibility, and interdisciplinary coherence in blind evaluation; compare against single-domain SciAgents "), bold("[P2]"), run(" baseline; require >80% of hypotheses to be provenance-traceable.")),

      sp(),

      // ═══════════════════════════════════════════════════════════════════════
      // SECTION 4 — TAXONOMY (BONUS)
      // ═══════════════════════════════════════════════════════════════════════
      h1("4. Taxonomy of LLM Agents for Scientific Research  (Bonus)"),

      p(run("The taxonomy below organises all 10 corpus papers into a 2-level hierarchy. The root is "), bold('"LLM Agents for Scientific Research."'), run(" Three Level-1 category nodes branch from it; nine Level-2 leaf nodes each contain a definition and at least one [P#] citation. Across the taxonomy, all 10 corpus papers are cited.")),

      h3("A. Knowledge Encoding Approaches"),
      p(bold("A.1 Parametric Scientific Pretraining: "), run("Models that encode domain knowledge directly into weights via large-scale corpus training with domain-specific tokenization. Galactica "), bold("[P9]"), run(" uses 48M+ curated scientific documents with custom SMILES, LaTeX, and citation tokens, outperforming PaLM 540B on MATH with 18× fewer parameters. BioGPT "), bold("[P10]"), run(" uses 15M PubMed abstracts with an in-domain BPE vocabulary to achieve SOTA on five biomedical NLP tasks. Shared limitation: parametric knowledge becomes stale without retraining and cannot distinguish true scientific claims from training-data artefacts.")),
      p(bold("A.2 Symbolic Knowledge Graph Representations: "), run("Systems encoding domain knowledge as explicit ontological graphs with typed nodes and edges, enabling relational reasoning and traceable hypothesis provenance. SciAgents "), bold("[P2]"), run(" constructs a graph from ~1,000 biology papers with hierarchical subgraph sampling for multi-agent context. Advantage over parametric approaches: knowledge is updatable and each inference step is traceable. Limitation: coverage is bounded by corpus breadth and single-domain focus.")),

      h3("B. Agentic Reasoning and Tool Use"),
      p(bold("B.1 Self-Supervised Tool Learning: "), run("Paradigms that teach LMs to invoke external APIs without human annotation, using model-internal signals. Toolformer "), bold("[P5]"), run(" filters API calls via a perplexity-reduction criterion, enabling a 6.7B model to match 175B models on zero-shot benchmarks. Limitation: the filter optimises for training-data coverage rather than knowledge-gap coverage, underselecting tools most needed for novel scientific queries.")),
      p(bold("B.2 Interleaved Reasoning–Action Frameworks: "), run("Methods that augment the agent's action space with a language reasoning space, enabling dynamic plan updates from environmental observations. ReAct "), bold("[P6]"), run(" achieves 34% absolute improvement over RL baselines on ALFWorld. Limitation: the reported gains are confounded by base-model scale—ReAct uses PaLM-540B while RL baselines used smaller models.")),
      p(bold("B.3 Verbal Reinforcement Learning: "), run("Agents that improve across trials via natural language self-reflection stored in episodic memory, without gradient updates. Reflexion "), bold("[P7]"), run(" achieves 91% pass@1 on HumanEval using an Actor–Evaluator–Self-Reflection triad. Limitation: self-generated unit tests used as success signals may be incomplete, overestimating real-world accuracy.")),
      p(bold("B.4 Domain-Specific Tool Integration: "), run("Bespoke expert-curated tool suites assembled for targeted scientific task completion and integrated with a base LLM. ChemCrow "), bold("[P4]"), run(" provides 13 chemistry tools (retrosynthesis, safety screening, SMILES operations) to GPT-4. PaperQA "), bold("[P3]"), run(" provides modular tools for scientific paper retrieval, evidence extraction, and multi-document synthesis. Limitation: significant expert effort required; tools are domain-specific and do not transfer.")),

      h3("C. Evaluation Frameworks for Scientific AI"),
      p(bold("C.1 End-to-End Pipeline Evaluation: "), run("Full-pipeline systems that evaluate agents across the entire scientific research cycle—ideation, implementation, experimentation, paper writing, and peer review. The AI Scientist "), bold("[P1]"), run(" produces papers for under $15 each with a GPT-4o reviewer achieving 65% balanced accuracy. Critical limitation: circular evaluation design where reviewer and generated papers share the same ICLR distributional signature.")),
      p(bold("C.2 Benchmarked Research Task Evaluation: "), run("Structured benchmark suites targeting specific AI research skills with standardised metrics and controlled experimental conditions. MLGym "), bold("[P8]"), run(" provides 13 open-ended ML research tasks across computer vision, NLP, RL, and game theory, with a six-level capability hierarchy. Limitation: only Level 1 is empirically populated; Levels 2–5 remain theoretical and unfalsifiable.")),

      // FIGURE
      sp(),
      new Paragraph({
        children: [
          new ImageRun({
            data: figData,
            transformation: { width: 630, height: 300 },
            type: "png"
          })
        ],
        alignment: AlignmentType.CENTER,
        spacing: { before: 80, after: 40 }
      }),
      new Paragraph({
        children: [new TextRun({
          text: "Figure 1.  Taxonomy of LLM Agents for Scientific Research  (2-level hierarchy · 13 nodes · 10/10 corpus papers cited)",
          size: 19, font: "Arial", italics: true, color: "444444"
        })],
        alignment: AlignmentType.CENTER,
        spacing: { before: 20, after: 280 }
      }),

      // ═══════════════════════════════════════════════════════════════════════
      // SECTION 5 — REFLECTION
      // ═══════════════════════════════════════════════════════════════════════
      h1("5. Reflection vs. Comparison Survey [S2]"),

      p(run("The comparison survey is "), bold("S2: "), italic("Scientific Large Language Models: A Survey on Biological & Chemical Domains"), run(" (Zhang, Ding et al., 2024) "), bold("[S2]"), run(", a 90-page treatment of LLMs for molecular science, protein language models, genomic sequence models, and multi-modal scientific language.")),

      h3("What the Survey Covers That We Did Not"),
      p(bold("[S2]"), run(" provides extensive coverage of molecular and genomic representation learning—protein language models (ESM, ProteinBERT, ProtGPT2), genomic sequence models (DNABERT, Nucleotide Transformer), and molecule-specific architectures fusing graph neural networks with transformers—none of which appear in our corpus. Our corpus omits this representation-learning axis because our 10 papers prioritize agentic task-solving and end-to-end autonomy over molecular representation as a standalone goal. Galactica "), bold("[P9]"), run(" handles SMILES and proteins at the surface level, and BioGPT "), bold("[P10]"), run(" handles biomedical text, but neither addresses protein structure modeling or genomic sequence learning at the depth Zhang et al. survey. This omission is appropriate given the corpus constraint but would need to be addressed in any complete survey of scientific AI.")),

      h3("What We Cover That the Survey Underplays"),
      p(bold("[S2]"), run(" concentrates almost exclusively on static model architectures, pretraining strategies, and fine-tuning recipes. Agentic capabilities—tool use, multi-step reasoning, self-reflection, and autonomous experiment execution—receive negligible treatment. ReAct "), bold("[P6]"), run(", Reflexion "), bold("[P7]"), run(", Toolformer "), bold("[P5]"), run(", ChemCrow "), bold("[P4]"), run(", The AI Scientist "), bold("[P1]"), run(", and MLGym "), bold("[P8]"), run(" are absent from Zhang et al. Most consequentially, our analysis reveals a finding that a static-model survey structurally cannot surface: the "), bold("novelty gap"), run("—all frontier LLMs on MLGym-Bench remain at Level 1 capability, failing to generate algorithmic innovations. This agentic dimension is the most policy-relevant finding for anyone deploying scientific AI today, and it is entirely invisible to a survey focused on pretraining architectures.")),

      h3("One Evaluation Weakness Common to Both"),
      p(run("Both surveys rely on benchmark-based evaluation that cannot detect sophisticated training-data interpolation. Zhang et al. "), bold("[S2]"), run(" use protein folding benchmarks, drug-target interaction datasets, and chemical property prediction tasks—all with known ground truths reachable by memorisation. Our corpus uses HumanEval, HotpotQA, MLGym-Bench, and LitQA with equivalent vulnerability. As C5 and C6 demonstrate, The AI Scientist's reviewer and Galactica both achieve strong calibrated metrics while exhibiting qualitative failures (circular evaluation, deployed misinformation) those metrics were blind to. The shared gap: neither survey proposes an adversarial test to distinguish systems that have genuinely solved a scientific problem from systems that have produced sophisticated interpolations from training data.")),

      h3("One Concrete Improvement You Would Make Next"),
      p(run("The single highest-value improvement would be operationalizing Direction 1 as a mandatory evaluation protocol: for each claim in the Key Claims table, specify a minimum out-of-distribution novelty distance (embedding similarity to training corpus below a threshold), a minimum blind expert validation rate (≥2 independent domain experts confirming non-trivial contribution), and a minimum provenance-traceability score. This would make C4, C5, and C10 fully falsifiable—transforming them from descriptive findings into testable scientific claims with explicit quantitative success criteria—and would address the interpolation-blindness weakness shared with "), bold("[S2]"), run(".")),

      sp(),

      // ═══════════════════════════════════════════════════════════════════════
      // REFERENCES
      // ═══════════════════════════════════════════════════════════════════════
      h1("References"),

      plain("[P1] Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024). The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. arXiv:2408.06292."),
      plain("[P2] Ghafarollahi, A., & Buehler, M. J. (2024). SciAgents: Automating Scientific Discovery Through Multi-Agent Intelligent Graph Reasoning. arXiv:2409.05556."),
      plain("[P3] Lala, J., O'Donoghue, O., Shtedritski, A., Cox, S., Rodriques, S. G., & White, A. D. (2023). PaperQA: Retrieval-Augmented Generative Agent for Scientific Research. arXiv:2312.07559."),
      plain("[P4] Bran, A. M., Cox, S., White, A. D., & Schwaller, P. (2023). ChemCrow: Augmenting Large-Language Models with Chemistry Tools. arXiv:2304.05376."),
      plain("[P5] Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. arXiv:2302.04761."),
      plain("[P6] Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023. arXiv:2210.03629."),
      plain("[P7] Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv:2303.11366."),
      plain("[P8] Nathani, D., Madaan, L., Roberts, N., Bashlykov, N., Menon, A., Moens, V., et al. (2025). MLGym: A New Framework and Benchmark for Advancing AI Research Agents. arXiv:2502.14499."),
      plain("[P9] Taylor, R., Kardas, M., Cucurull, G., Scialom, T., Hartshorn, A., Saravia, E., Poulton, A., Kerkez, V., & Stojnic, R. (2022). Galactica: A Large Language Model for Science. arXiv:2211.09085."),
      plain("[P10] Luo, R., Sun, L., Xia, Y., Qin, T., Zhang, S., Poon, H., & Liu, T.-Y. (2022). BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining. Briefings in Bioinformatics, 23(6)."),
      plain("[S2] Zhang, Q., Ding, K., Lyu, T., Wang, X., Yin, Q., Zhang, Y., et al. (2024). Scientific Large Language Models: A Survey on Biological & Chemical Domains. arXiv:2401.14656."),
    ]
  }]
});

Packer.toBuffer(doc).then(buf => {
  const outPath = path.join(__dirname, '..', 'paper.docx');
  fs.writeFileSync(outPath, buf);
  console.log(`paper.docx written ✓  ${(buf.length/1024).toFixed(0)} KB → ${outPath}`);
}).catch(e => { console.error(e); process.exit(1); });
