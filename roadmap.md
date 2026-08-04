# Roadmap to a More Auditable, Evaluated, and Enterprise-Ready CDC Refinement System

## 1. Objective

The current system is already a strong prototype: it combines structured state management, gap detection, RAG-based resolution, human-in-the-loop clarification, contradiction detection, persistent execution, and final document generation.

The next step should not be to make the system more complex.

The goal should be to make it:

1. **More auditable** — every important decision can be traced back to its origin.
2. **More measurable** — the quality of each AI component can be evaluated quantitatively.
3. **More reproducible** — the same CDC and configuration can be replayed and compared.
4. **More autonomous during testing** — human answers can be simulated automatically for batch evaluation.
5. **More complete as a product** — the final output should contain not only a polished CDC, but also traceability, risks, assumptions, unresolved gaps, and quality metrics.
6. **More convincing to an enterprise** — the system should demonstrate measurable value rather than only technical sophistication.

The recommended target architecture is:

```text
                    Initial CDC
                         │
                         ▼
                Gap Detection Layer
                         │
                         ▼
              RAG / Evidence Retrieval
                         │
              ┌──────────┴──────────┐
              │                     │
         Evidence found        No evidence
              │                     │
              ▼                     ▼
        AI validation          Question
              │                     │
              └──────────┬──────────┘
                         ▼
                  Human / Simulator
                         │
                         ▼
                Context Integration
                         │
                         ▼
              Contradiction Detection
                         │
                         ▼
                Iterative Refinement
                         │
                         ▼
                 Final Validation
                         │
                         ▼
              ┌──────────┴──────────┐
              ▼                     ▼
         Final CDC             QA Package
                                    │
                         ┌──────────┼──────────┐
                         ▼          ▼          ▼
                    Traceability  Metrics    Risks
```

The key idea is to evolve from:

> "An AI system that generates a better CDC"

toward:

> "An auditable requirements intelligence system that can demonstrate why the final CDC is complete, consistent, and trustworthy."

---

# 2. Priority roadmap

Do not implement everything at once.

Recommended order:

| Priority | Addition                           |     Effort |    Impact |
| -------- | ---------------------------------- | ---------: | --------: |
| P0       | Evaluation dataset + ground truth  |     Medium | Very High |
| P0       | Automated HIL batch simulator      | Low/Medium | Very High |
| P0       | Audit trail / provenance           |     Medium | Very High |
| P1       | ML-style evaluation pipeline       |     Medium | Very High |
| P1       | Improved final QA report           |     Medium | Very High |
| P1       | Requirement traceability matrix    |     Medium |      High |
| P1       | Confidence / risk scoring          |     Medium |      High |
| P2       | RAG retrieval evaluation           |     Medium |      High |
| P2       | Experiment tracking                |     Medium |      High |
| P2       | Regression testing                 |     Medium |      High |
| P3       | Enterprise integrations            |       High |      High |
| P3       | Authentication / RBAC / governance |       High |      High |

For a short internship, I would focus primarily on **P0 + P1**.

---

# 3. Improve auditability first

## 3.1 Add provenance to every ContextItem

The current system already has a good foundation: `ContextItem` records whether information came from the initial CDC, a user answer, RAG, or an assumption.

Extend this model.

Instead of:

```python
class ContextItem:
    id
    content
    source
    section_ids
    linked_gap_id
    turn_added
    fresh
```

move toward:

```python
class ContextItem:
    id
    content

    source
    source_document
    source_page
    source_chunk_id

    section_ids
    linked_gap_id

    created_by
    turn_added
    timestamp

    confidence
    validation_status
```

Possible values:

```text
source:
    initial_cdc
    user_answer
    rag
    assumption

created_by:
    user
    rag
    llm
    system

validation_status:
    unreviewed
    accepted
    rejected
    needs_review
```

This immediately answers:

> Where did this information come from?

---

## 3.2 Give RAG answers explicit evidence

Instead of storing only:

```text
source = "rag"
content = "The manager can modify stock."
```

store:

```text
ContextItem
├── content
├── source = rag
├── document = stock_process.pdf
├── page = 14
├── chunk_id = stock_process::42
├── retrieval_score = 0.87
├── evidence_grade = sufficient
└── confidence = high
```

The final CDC can then reference the evidence.

The user should be able to click or inspect:

```text
FR-023
"The store manager can modify stock quantities."

Evidence:
[stock_process.pdf, page 14]

Origin:
RAG

Related gap:
GAP-042

Validation:
Automatically accepted
```

This is much more enterprise-friendly.

---

## 3.3 Add a decision log

Your current `turn_log` is already an append-only history for the UI.

Extend the idea into an explicit decision log.

For every meaningful AI decision, record:

```text
timestamp
run_id
turn
agent
decision_type
input_ids
output
model
prompt_version
confidence
evidence_ids
```

Example:

```json
{
  "run_id": "run_123",
  "turn": 4,
  "agent": "gap_filler",
  "decision_type": "rag_answer",
  "gap_id": "gap_042",
  "evidence_ids": ["doc_12_chunk_42"],
  "confidence": 0.91,
  "model": "gpt-oss:20b",
  "prompt_version": "gap_filler_v3"
}
```

This allows you to answer:

> Why did the system make this decision?

---

# 4. Add a traceability graph

This is one of the highest-value additions.

The system should be able to represent:

```text
Source Document
      │
      ▼
ContextItem
      │
      ▼
Gap
      │
      ▼
Answer / Evidence
      │
      ▼
Requirement
      │
      ▼
Acceptance Criterion
      │
      ▼
Final CDC Section
```

For example:

```text
DOC-014
   │
   ▼
CTX-087
   │
   ▼
GAP-032
   │
   ▼
Q-019
   │
   ▼
USER-ANSWER-019
   │
   ▼
REQ-F-012
   │
   ├── AC-012-01
   ├── AC-012-02
   └── BR-007
```

This creates a traceability matrix.

Example:

| Requirement | Source            | Gap     | Resolution | Acceptance Criteria | Status       |
| ----------- | ----------------- | ------- | ---------- | ------------------- | ------------ |
| FR-012      | User answer Q-019 | GAP-032 | Human      | AC-012-01           | Validated    |
| FR-013      | Reference PDF     | GAP-041 | RAG        | AC-013-01           | Validated    |
| FR-014      | Assumption        | GAP-052 | Assumption | Missing             | Needs review |

This is significantly more compelling than simply generating a DOCX.

---

# 5. Treat evaluation like an ML project

This is probably the most important improvement.

The current architecture is strong, but the system needs measurable performance.

Do not evaluate the entire system with one subjective score.

Break it into components.

```text
System Evaluation
│
├── Gap Detection
├── RAG Retrieval
├── RAG Answer Validation
├── Question Generation
├── Contradiction Detection
├── Assumption Generation
├── Final Requirement Quality
└── End-to-End CDC Quality
```

Each component should have its own evaluation.

---

# 6. Build a benchmark dataset

Create a small but carefully annotated dataset.

For example:

```text
20 CDCs
```

Each CDC should have:

```text
Initial CDC
Reference Documents
Ground Truth Gaps
Ground Truth Contradictions
Expected Answers
Expected Questions
Final Expert-Validated CDC
```

A simple structure:

```text
evaluation/
    datasets/
        cdc_001/
            initial_cdc.md
            source_docs/
            ground_truth.json
            expected_final.md

        cdc_002/
            initial_cdc.md
            source_docs/
            ground_truth.json
            expected_final.md
```

The ground truth could contain:

```json
{
  "gaps": [
    {
      "description": "User roles are not defined",
      "category": "functional_ambiguity",
      "severity": "important"
    }
  ],
  "contradictions": [
    {
      "statement_a": "...",
      "statement_b": "..."
    }
  ]
}
```

The ground truth does not need to be perfect.

It should be created or validated by a human expert.

---

# 7. Evaluate gap detection like a classification problem

For each expected gap:

```text
Ground Truth Gap
        │
        ├── Detected → True Positive
        └── Missed    → False Negative
```

For each system-generated gap:

```text
Generated Gap
        │
        ├── Valid → True Positive
        └── Invalid → False Positive
```

Measure:

```text
Precision
Recall
F1-score
```

Evaluate by category:

```text
functional_ambiguity
nfr
data_model
business_rule
edge_case
integration
acceptance_criteria
contradiction
scope
```

This is much more informative than:

> "The system found 23 gaps."

You want:

> "The system achieved 82% recall and 78% precision for blocking/important gaps."

---

# 8. Evaluate severity classification

The system also predicts:

```text
blocking
important
nice_to_have
```

Create a confusion matrix.

Example:

| Actual / Predicted | Blocking | Important | Nice |
| ------------------ | -------: | --------: | ---: |
| Blocking           |       18 |         2 |    0 |
| Important          |        3 |        14 |    3 |
| Nice               |        0 |         4 |   16 |

This reveals whether the system is incorrectly treating critical gaps as minor.

For enterprise use, **false negatives on blocking gaps** are more dangerous than false positives.

Therefore, introduce a metric:

> **Blocking Gap Recall**

This could become one of your primary safety metrics.

---

# 9. Evaluate contradiction detection separately

Create a test set containing:

```text
True contradictions
Non-contradictory statements
Subtle contradictions
Cross-section contradictions
```

Measure:

```text
Contradiction Precision
Contradiction Recall
Contradiction F1
```

Also measure:

> **Critical contradiction recall**

The system should prioritize not missing serious contradictions.

---

# 10. Evaluate RAG as a retrieval system

The RAG system currently retrieves the top-k documents based on semantic similarity.

Treat it like a real information retrieval system.

For every gap that should be answerable from the reference documents, annotate:

```text
Expected document
Expected section
Expected chunk
```

Measure:

```text
Recall@1
Recall@3
Recall@5
MRR
```

Example:

```text
Recall@1 = 62%
Recall@3 = 84%
Recall@5 = 92%
```

This tells you whether your retrieval is actually working.

Then separately evaluate:

> Did the LLM correctly determine whether the retrieved evidence was sufficient?

This separates:

```text
Retrieval problem
```

from:

```text
Reasoning problem
```

That distinction is important.

---

# 11. Evaluate the question generator

The question generator should not be judged only by whether the question "sounds good."

Create evaluation criteria:

```text
Question Quality
├── Addresses the correct gap
├── Is specific
├── Is understandable
├── Contains enough context
├── Does not duplicate previous questions
└── Is answerable by the stakeholder
```

Score each dimension:

```text
0 = bad
1 = acceptable
2 = good
```

You can calculate:

```text
Average Question Quality Score
```

Also measure:

> **Question efficiency**

For example:

```text
How many questions are required
to resolve one important gap?
```

Your goal should be to minimize:

```text
Questions per resolved gap
```

without sacrificing completeness.

---

# 12. Evaluate end-to-end improvement

Ultimately, the most important question is:

> Does the final CDC become better?

Create:

```text
Initial CDC
      │
      ▼
Human expert score
```

Then:

```text
Final CDC
      │
      ▼
Human expert score
```

Score dimensions:

```text
Completeness
Consistency
Clarity
Testability
Traceability
Business-rule coverage
NFR coverage
Acceptance criteria quality
```

Example:

| Metric       | Initial | Final |
| ------------ | ------: | ----: |
| Completeness |     48% |   91% |
| Consistency  |     62% |   94% |
| Testability  |     38% |   83% |
| Traceability |     21% |   89% |

This is your strongest evidence of value.

---

# 13. Automate HIL for batch testing

You asked for a quick solution to automate human answers.

The easiest approach is to create a **Synthetic Stakeholder Simulator**.

Instead of:

```text
Graph
  ↓
interrupt()
  ↓
Human
```

during evaluation:

```text
Graph
  ↓
interrupt()
  ↓
Synthetic Stakeholder
  ↓
Command(resume=answers)
  ↓
Graph resumes
```

The simulator should be driven by a predefined "stakeholder profile."

For example:

```yaml
stakeholder:
  name: "Stock Manager"
  role: "Business Owner"

knowledge:
  - "Store managers can modify stock."
  - "Orders above 10,000 MAD require central approval."
  - "Alerts are automatically closed after 3 days."

unknown:
  - "Exact SLA for supplier response"

rules:
  - If asked about stock modification:
      answer: "Store managers can modify stock."

  - If asked about approval:
      answer: "Orders above 10,000 MAD require central approval."

  - If asked about unknown information:
      answer: "I don't know."
```

The simulator receives the actual question:

```text
Question:
"Which roles can modify stock quantities?"
```

Then returns:

```text
"Store managers can modify stock quantities."
```

This is much better than simply hardcoding answers by question ID.

---

# 14. Use an LLM-based synthetic stakeholder for scale

For more realistic testing:

```text
Reference Documents
        │
        ▼
Stakeholder Profile
        │
        ▼
Synthetic Stakeholder LLM
        │
        ▼
Answer generated from allowed knowledge
```

The simulator prompt should enforce:

> You may only answer using information contained in the provided stakeholder knowledge. If the information is not available, answer "I don't know."

This is critical.

Otherwise the test simulator will hallucinate answers and artificially inflate system performance.

You can also simulate different stakeholder types:

```text
Business Owner
Technical Architect
Security Officer
Store Manager
Product Manager
```

Each has a different knowledge base.

---

# 15. Run two evaluation modes

You should have:

### Mode A — Oracle answers

The simulator always provides the known ground-truth answer.

Purpose:

> Measure whether the system can correctly integrate correct information.

### Mode B — Realistic stakeholder

The simulator sometimes:

* doesn't know
* gives incomplete answers
* gives ambiguous answers
* contradicts earlier information

Purpose:

> Measure whether the system behaves correctly under realistic interaction.

This is especially useful for testing your contradiction and assumption mechanisms.

---

# 16. Add a deterministic batch runner

Create something like:

```text
python evaluate.py
```

Conceptually:

```python
for dataset in benchmark:
    state = run_system(dataset.initial_cdc)

    while not state.done:
        if interrupted:
            answers = simulator.answer(
                questions,
                stakeholder_profile
            )

            state = resume_graph(answers)

    evaluate(
        predictions=state,
        ground_truth=dataset.ground_truth
    )
```

Output:

```text
evaluation/results/run_2026_07_28.json
evaluation/results/summary.csv
evaluation/reports/report.md
```

This turns your project into an actual experimental framework.

---

# 17. Make experiments reproducible

Every evaluation run should record:

```text
dataset version
model
model version
temperature
embedding model
chunk size
top_k
prompt version
sections.yaml version
settings.yaml version
git commit
```

For example:

```text
Experiment: EXP-042

Dataset: benchmark_v2
LLM: gpt-oss:20b
Embedding: nomic-embed-text-v2-moe
Temperature: 0
RAG top_k: 4
Prompt version: v7
Git commit: a81f23c
```

This allows you to compare:

```text
Experiment A
vs
Experiment B
```

scientifically.

---

# 18. Add regression testing

Once you have the benchmark, every code or prompt modification should run it.

Example:

```text
Before change:
Gap F1 = 0.78
Contradiction F1 = 0.81
RAG Recall@5 = 0.89

After change:
Gap F1 = 0.83
Contradiction F1 = 0.79
RAG Recall@5 = 0.91
```

Now you know:

> The change improved gap detection but degraded contradiction detection.

This is much more mature than manually testing a few CDCs.

---

# 19. Improve the final output

The final output should become a **QA package**, not just a DOCX.

I recommend generating:

```text
output/
    cdc_final.docx
    cdc_final.qmd

    qa_report.md
    traceability_matrix.xlsx
    requirements.json
    unresolved_gaps.json
    assumptions.json
    evidence_map.json
```

The user can then choose the level of detail.

---

# 20. Improve the final CDC itself

The final document should ideally contain:

```text
1. Executive Summary

2. Problem & Context

3. Objectives and KPIs

4. Scope
   - In scope
   - Out of scope

5. Stakeholders and Roles

6. Functional Requirements

7. Business Rules

8. Data Requirements

9. Non-Functional Requirements

10. Integrations

11. Security and Access Control

12. Constraints

13. Edge Cases

14. Acceptance Criteria

15. Assumptions

16. Open Questions

17. Dependencies

18. Risks

19. Traceability
```

This makes the final CDC much more useful to a development team.

---

# 21. Add requirement IDs

Every requirement should have a stable identifier.

Example:

```text
FR-001
FR-002
BR-001
NFR-001
AC-001
```

Then you can create relationships:

```text
FR-001
  ├── BR-001
  ├── BR-002
  ├── AC-001
  └── AC-002
```

This makes the document testable and maintainable.

---

# 22. Add a final quality score

Generate an executive QA summary:

```text
CDC QUALITY SCORE
=================

Completeness       89%
Consistency         94%
Traceability       87%
Testability        82%
NFR Coverage       76%

Blocking gaps       0
Important gaps      2
Nice-to-have gaps   5

Unvalidated assumptions: 3

Overall status:
⚠️ READY WITH CONDITIONS
```

Possible statuses:

```text
NOT READY
READY WITH CONDITIONS
DEVELOPMENT READY
```

The status should be determined by deterministic rules.

For example:

```text
DEVELOPMENT READY if:
    blocking gaps = 0
    important gaps <= 2
    critical contradictions = 0
    unvalidated high-impact assumptions = 0
```

This is better than allowing an LLM to simply declare:

> "The document is complete."

---

# 23. Separate AI judgment from deterministic validation

This is extremely important.

Use the LLM for:

```text
Semantic gap detection
Semantic contradiction detection
Question generation
Text synthesis
```

Use deterministic code for:

```text
Required sections present
Blocking gaps count
Requirement IDs
Missing acceptance criteria
Unresolved assumptions
Traceability coverage
Output files
Schema validation
```

The architecture should look like:

```text
             LLM
              │
       Semantic reasoning
              │
              ▼
        Structured output
              │
              ▼
     Deterministic validation
              │
              ▼
        Final QA decision
```

This makes the system more trustworthy.

---

# 24. Add confidence and risk

Every AI-generated artifact should ideally have:

```text
confidence
risk
validation_required
```

Example:

```text
RAG Answer
Confidence: 0.91
Risk: Low

Assumption
Confidence: 0.42
Risk: High
Validation required: Yes
```

Do not necessarily treat confidence as a true probability.

Instead, use it as an operational score.

For example:

```text
0.80–1.00 → High confidence
0.50–0.79 → Medium
0.00–0.49 → Low
```

Then define policies:

```text
High confidence + low risk
→ automatic acceptance

Medium confidence
→ QA review

Low confidence or high risk
→ human validation required
```

---

# 25. Add a "risk register"

The system should identify:

```text
Risk
├── Description
├── Source
├── Impact
├── Likelihood
├── Severity
├── Mitigation
└── Owner
```

Example:

```text
RISK-003

Description:
Supplier API availability is not specified.

Source:
Integration gap GAP-042

Impact:
High

Likelihood:
Medium

Mitigation:
Define fallback behavior.

Status:
Open
```

This moves the product beyond document generation toward actual project preparation.

---

# 26. Add a "human effort saved" metric

This is critical for enterprise value.

Measure:

```text
Questions that could be answered by RAG
```

versus:

```text
Questions actually sent to humans
```

Calculate:

```text
Human Intervention Reduction
```

For example:

```text
Total gaps: 100

Resolved by RAG: 35
Resolved by human: 50
Assumed/deferred: 15

Human intervention reduction:
35%
```

This directly demonstrates the value of RAG.

Also measure:

```text
Average questions per CDC
Average turns per CDC
Average time to completion
```

---

# 27. Your final product should have three outputs

I would define the product as:

### 1. The CDC

The document for the development team.

### 2. The QA report

The document for the project manager / business analyst.

### 3. The evidence and traceability package

The artifact for auditors, architects, and technical reviewers.

Conceptually:

```text
                    Final Run
                       │
           ┌───────────┼───────────┐
           ▼           ▼           ▼
          CDC        QA Report    Evidence
           │           │           │
       Developers    Managers    Auditors
```

This is much more compelling than a single DOCX.

---

# 28. Recommended implementation plan

## Phase 1 — Auditability

Implement:

```text
[x] Extend ContextItem provenance
[x] Store source document/page/chunk
[x] Add confidence
[x] Add validation status
[x] Add decision log
[x] Add prompt/model/version metadata
```

Result:

> Every AI decision becomes explainable.

**Done.** See `src/state.py` (`Evidence`, `DecisionLogEntry`, extended
`ContextItem`), `src/decisions.py` (confidence policy, evidence/decision
builders), `src/prompts.py` (prompt version registry), page-aware chunking in
`src/rag.py`, and `output/decision_log.jsonl`. The UI surfaces evidence and
confidence on gap cards, plus a decision-log table under "Sous le capot".

---

## Phase 2 — Evaluation dataset

Implement:

```text
[ ] 10–20 benchmark CDCs
[ ] Ground truth gaps
[ ] Ground truth contradictions
[ ] Ground truth source documents
[ ] Expert-validated expected final output
```

Result:

> You have something measurable.

---

## Phase 3 — Automated HIL

Implement:

```text
[ ] Stakeholder profiles
[ ] Knowledge base per stakeholder
[ ] Synthetic answer simulator
[ ] "I don't know" behavior
[ ] Contradictory-answer scenarios
[ ] Batch graph execution
```

Result:

> You can run hundreds of experiments without manually answering questions.

---

## Phase 4 — ML-style evaluation

Implement:

```text
[ ] Gap precision / recall / F1
[ ] Blocking-gap recall
[ ] Contradiction precision / recall / F1
[ ] RAG Recall@K
[ ] MRR
[ ] Question quality score
[ ] Human intervention reduction
[ ] End-to-end completeness improvement
```

Result:

> The system becomes scientifically evaluated.

---

## Phase 5 — Final output

Implement:

```text
[ ] Requirement IDs
[ ] Acceptance criteria
[ ] Assumptions
[ ] Open questions
[ ] Risks
[ ] Traceability matrix
[ ] Evidence map
[ ] Quality score
[ ] Development readiness status
```

Result:

> The output becomes useful to a real organization.

---

## Phase 6 — Regression framework

Implement:

```text
[ ] Evaluation CLI
[ ] Experiment IDs
[ ] Dataset versioning
[ ] Prompt versioning
[ ] Model versioning
[ ] JSON/CSV result storage
[ ] Comparison reports
```

Result:

> You can demonstrate that changes actually improve the system.

---

# 29. What I would personally prioritize for your internship

Given your time constraints, I would **not** try to implement enterprise SSO, Jira integration, Kubernetes deployment, or a huge production infrastructure.

I would focus on this:

```text
                    CURRENT SYSTEM
                         │
                         ▼
              ┌───────────────────┐
              │ 1. Auditability   │
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │ 2. Benchmark      │
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │ 3. Auto-HIL       │
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │ 4. ML Evaluation  │
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │ 5. QA + Traceability
              └─────────┬─────────┘
                        ▼
               COMPELLING PRODUCT
```

If you complete only these five areas, the project changes significantly.

Instead of presenting:

> "I built a multi-agent CDC generator."

You can present:

> "I built and evaluated an auditable AI-assisted requirements engineering system. I created a benchmark dataset with ground truth annotations, automated the human-in-the-loop evaluation process using synthetic stakeholders, measured gap detection and contradiction detection with precision/recall/F1, evaluated RAG retrieval with Recall@K, measured the reduction in human intervention, and generated a final requirements document with evidence traceability, risk analysis, assumptions, and quality scoring."

That is a **much stronger engineering-school project** and a much more credible enterprise product.

---

# 30. Final target architecture

The final vision I would aim for is:

```text
┌───────────────────────────────────────────────────────────────┐
│                    REQUIREMENTS INTELLIGENCE PLATFORM          │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  INPUT                                                        │
│  ├── Initial CDC                                               │
│  ├── Reference Documents                                       │
│  └── Existing Enterprise Knowledge                             │
│                                                               │
│                         ▼                                     │
│                                                               │
│  INTELLIGENCE                                                  │
│  ├── Gap Detection                                             │
│  ├── RAG Retrieval                                             │
│  ├── Evidence Validation                                       │
│  ├── Targeted Questions                                        │
│  └── Contradiction Detection                                   │
│                                                               │
│                         ▼                                     │
│                                                               │
│  HUMAN-IN-THE-LOOP                                             │
│  ├── Human Answers                                             │
│  ├── Assumption Validation                                     │
│  └── Risk Review                                               │
│                                                               │
│                         ▼                                     │
│                                                               │
│  QUALITY ENGINE                                                │
│  ├── Completeness                                              │
│  ├── Consistency                                               │
│  ├── Traceability                                              │
│  ├── Testability                                               │
│  └── Readiness Score                                           │
│                                                               │
│                         ▼                                     │
│                                                               │
│  OUTPUT                                                        │
│  ├── Development-ready CDC                                    │
│  ├── QA Report                                                 │
│  ├── Traceability Matrix                                       │
│  ├── Evidence Map                                              │
│  ├── Risk Register                                              │
│  └── Open Issues / Assumptions                                 │
│                                                               │
│                         ▼                                     │
│                                                               │
│  EVALUATION & OBSERVABILITY                                    │
│  ├── Precision / Recall / F1                                   │
│  ├── RAG Recall@K                                              │
│  ├── Human Effort Reduction                                    │
│  ├── Latency / Cost                                            │
│  ├── Experiment Tracking                                       │
│  └── Regression Testing                                        │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## The main principle

**Do not add complexity just to make the architecture look more advanced.**

Your current architecture is already advanced enough.

The next stage is to make it **provable**.

You want to be able to demonstrate:

> **"Here is what the system detected."**

> **"Here is what it retrieved."**

> **"Here is why it trusted the evidence."**

> **"Here is what it asked the human."**

> **"Here is how the answer changed the requirements."**

> **"Here is the contradiction it detected."**

> **"Here is how we measured whether it was correct."**

> **"Here is how much human effort it saved."**

> **"Here is why we consider the final CDC development-ready."**

That is the transition from an impressive **multi-agent prototype** to a compelling **AI product with engineering rigor and enterprise potential**.
