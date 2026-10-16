# Student Voice Intelligence
## 8-Day Portfolio Project Plan

### Project Goal

Build a production-oriented **LLM-assisted thematic analysis system for student feedback** that demonstrates:

- NLP and thematic analysis
- classical ML baselines
- embeddings and clustering
- BERTopic / topic discovery
- LLM-assisted theme generation and classification
- rigorous model and LLM evaluation
- human-in-the-loop review
- production architecture using FastAPI and Docker
- monitoring, versioning and continuous improvement

The project should answer the core question:

> **Can NLP and LLM-assisted thematic analysis reliably transform large volumes of unstructured student feedback into stable, interpretable and actionable themes while maintaining human oversight?**

The final project should demonstrate not only that the system works, but also **why particular methods were selected, how they were evaluated, where they fail, and how they could be operated safely in production.**

---

# Day 1 — Data Understanding, Quality and EDA

## Goal

Understand the student-feedback dataset before applying any machine-learning or NLP method.

The objective is to determine:

- what the dataset contains
- whether the labels and categories are balanced
- whether feedback is single-theme or multi-theme
- what data-quality problems exist
- what privacy or PII risks exist
- whether the current taxonomy is suitable for modelling

## Tasks

1. Load the public student-feedback dataset.
2. Inspect:
   - number of comments
   - available columns
   - categories
   - sentiment labels
3. Check:
   - missing values
   - empty comments
   - duplicate comments
   - extremely short responses
   - possible PII
4. Analyse comment length distribution.
5. Analyse category distribution.
6. Analyse sentiment distribution.
7. Identify comments that potentially contain multiple ideas.
8. Explore semantic overlap between existing categories using TF-IDF similarity.
9. Save a cleaned dataset for subsequent modelling.

## Questions to Answer

- Are categories imbalanced?
- Are some categories very similar?
- How common are short or low-information responses?
- Do comments frequently discuss more than one theme?
- Is single-label sentiment sufficient?
- What data should be excluded or flagged?

## Deliverables

```text
01_data_quality_and_eda.ipynb
or
day1_student_feedback_eda.py

day1_outputs/
├── cleaned_feedback.csv
├── data_quality_summary.md
├── category_distribution.csv
├── sentiment_distribution.csv
├── duplicate_comments.csv
├── very_short_comments.csv
├── possible_multi_idea_comments.csv
├── pii_review_queue.csv
└── plots/
```

## Key Learning Outcome

Be able to explain:

> Why free-text student feedback is a more complex analytical problem than simple sentiment classification.

---

# Day 2 — Classical NLP and ML Baselines

## Goal

Establish simple, interpretable baselines before introducing embeddings or LLMs.

The purpose is to answer:

> **How well can traditional NLP methods solve the problem before we add more complex AI approaches?**

## Tasks

Build three baseline approaches.

### Baseline 1

```text
TF-IDF
    ↓
Logistic Regression
    ↓
Feedback category prediction
```

### Baseline 2

```text
TF-IDF
    ↓
Logistic Regression
    ↓
Sentiment classification
```

### Baseline 3

```text
TF-IDF
    ↓
KMeans
    ↓
Unsupervised topic discovery
```

## Evaluation

For supervised models:

- precision
- recall
- macro-F1
- confusion matrix

For clustering:

- silhouette score
- cluster size distribution
- qualitative interpretability

## Questions to Answer

- How well do lexical features identify themes?
- Which categories are frequently confused?
- Does class imbalance affect performance?
- Can unsupervised clustering recover meaningful themes?
- What limitations justify moving to semantic embeddings?

## Deliverables

```text
02_classical_nlp_baselines.ipynb

results/
├── baseline_classification_metrics.csv
├── confusion_matrix.png
├── tfidf_cluster_summary.csv
└── baseline_findings.md
```

## Key Learning Outcome

Be able to justify why a more complex method is necessary instead of defaulting immediately to an LLM.

---

# Day 3 — Semantic Embeddings and Clustering

## Goal

Determine whether semantic representations improve theme discovery compared with lexical TF-IDF features.

## Tasks

Generate sentence embeddings using a SentenceTransformer model.

Compare multiple clustering pipelines.

### Approach A

```text
Sentence Embeddings
       ↓
PCA
       ↓
KMeans
```

### Approach B

```text
Sentence Embeddings
       ↓
UMAP
       ↓
HDBSCAN
```

Explore different:

- embedding models
- PCA dimensions
- UMAP `n_neighbors`
- UMAP dimensions
- HDBSCAN minimum cluster sizes
- random seeds

## Evaluation

Measure:

- silhouette score
- cluster size
- noise / unassigned percentage
- cluster interpretability
- stability across parameter changes

## Questions to Answer

- Are embedding-based clusters more coherent than TF-IDF clusters?
- How sensitive are the clusters to UMAP parameters?
- Does PCA produce more stable clusters?
- What happens when the random seed changes?
- Which method is appropriate for production vs visualisation?

## Deliverables

```text
03_embeddings_and_clustering.ipynb

results/
├── embedding_clusters.csv
├── cluster_metrics.csv
├── pca_visualisation.png
├── umap_visualisation.png
└── clustering_comparison.md
```

## Key Learning Outcome

Be able to clearly explain:

- PCA
- UMAP
- t-SNE
- when each should be used
- why dimensionality reduction can influence downstream clustering

---

# Day 4 — BERTopic and Theme Taxonomy Design

## Goal

Build an interpretable unsupervised thematic-analysis pipeline.

## Tasks

Apply BERTopic to the student-feedback corpus.

Generate for each topic:

- top keywords
- representative comments
- topic frequency
- topic description

Example:

```text
Topic 4

Keywords:
assessment
assignment
deadline
workload
submission

Representative comments:
1. ...
2. ...
3. ...

Proposed Theme:
Assessment workload and deadlines
```

Experiment with different levels of theme granularity.

For example:

```text
8 themes
12 themes
15 themes
20 themes
```

Develop a candidate taxonomy such as:

```text
Teaching Quality
Assessment Design
Assessment Workload
Feedback and Marking
Course Content
Learning Resources
Course Organisation
Technology
Student Support
Practical Activities
```

## Evaluation

Compare:

- topic coherence
- interpretability
- cluster size
- topic overlap
- stability

## Questions to Answer

- How many themes are useful?
- When do topics become too broad?
- When do topics become too fragmented?
- Can related themes be merged?
- Does the discovered taxonomy align with existing labels?

## Deliverables

```text
04_bertopic_theme_discovery.ipynb

results/
├── topic_summary.csv
├── representative_comments.csv
├── theme_taxonomy_v1.yaml
└── theme_taxonomy_analysis.md
```

## Key Learning Outcome

Understand the distinction between:

```text
topic modelling
vs
thematic analysis
vs
classification
```

---

# Day 5 — LLM-Assisted Thematic Analysis

## Goal

Use LLMs only where they provide measurable additional value.

The LLM should not simply analyse the entire dataset directly.

Instead, use it for controlled tasks.

## Task A — Theme Naming

Provide the LLM with:

- cluster keywords
- representative comments

Request:

```json
{
  "theme": "Assessment workload",
  "description": "Comments concerning assignment volume, deadlines and workload.",
  "confidence": 0.89
}
```

---

## Task B — Multi-Label Theme Classification

Example input:

```text
"The lecturer explained concepts clearly but the assignment workload was excessive."
```

Expected output:

```text
Teaching Quality → Positive
Assessment Workload → Negative
```

---

## Task C — Grounded Theme Summaries

Generate summaries only from actual supporting comments.

Store:

```text
theme
summary
supporting_comment_ids
```

This ensures every summary remains auditable.

## Tasks

1. Create structured prompts.
2. Force structured outputs.
3. Add evidence IDs.
4. Handle invalid model responses.
5. Record prompt and model version.
6. Record latency and cost if applicable.

## Deliverables

```text
05_llm_thematic_analysis.ipynb

src/llm/
├── classify_theme.py
├── name_theme.py
├── summarise_theme.py
└── prompts.py

results/
├── llm_theme_assignments.csv
├── grounded_summaries.csv
└── llm_failure_examples.md
```

## Key Learning Outcome

Be able to explain:

> Where does the LLM add genuine value compared with classical NLP?

---

# Day 6 — Evaluation Framework

## Goal

Build a rigorous evaluation system rather than relying on visually convincing LLM outputs.

This is one of the most important days of the project.

## Create a Gold Dataset

Manually label approximately:

```text
100–150 comments
```

For each comment, record:

```text
comment_id
theme(s)
sentiment
ambiguous?
multi-theme?
review_notes
```

Consider double-labelling 30–50 comments to assess human annotation consistency.

---

## Evaluation 1 — Theme Classification

Measure:

- precision
- recall
- macro-F1
- per-theme F1

For multi-label classification:

- micro-F1
- macro-F1
- Hamming loss

---

## Evaluation 2 — Clustering Stability

Compare different model runs using:

- Adjusted Rand Index
- Normalised Mutual Information

Also inspect qualitative consistency.

---

## Evaluation 3 — LLM Reliability Test Suite

Create approximately 30 dedicated test cases.

Include:

```text
clear single-theme comment
multi-theme comment
mixed sentiment
very short response
sarcasm
ambiguous comment
off-topic response
PII-like content
rare theme
contradictory feedback
```

Example:

```text
"Amazing lecturer. Worst assessment design I have experienced."
```

Correct:

```text
Teaching Quality → Positive
Assessment Design → Negative
```

Incorrect:

```text
Overall Sentiment → Neutral
```

---

## Evaluation 4 — Groundedness

Check whether generated summaries are supported by the source comments.

Evaluate:

- unsupported claims
- hallucinations
- citation/support accuracy

## Deliverables

```text
evaluation/
├── gold_labels.csv
├── reliability_test_cases.yaml
├── benchmark.py
└── evaluation_report.md
```

Produce a summary such as:

```text
Theme classification macro-F1     0.82
Multi-label micro-F1              0.86
Cluster stability ARI             0.74
Grounded summary accuracy         92%
LLM test-case success             88%
```

## Key Learning Outcome

Be able to explain:

> How do we know that the AI-generated themes are trustworthy?

---

# Day 7 — Production Architecture

## Goal

Move the project from notebook experimentation toward a production-style analytical system.

## Architecture

```text
Student Feedback
       ↓
Data Validation
       ↓
Text Preprocessing
       ↓
Embedding Service
       ↓
 ┌───────────────┐
 ↓               ↓
Theme        Novel Theme
Classifier   Discovery
 ↓               ↓
 └───────┬───────┘
         ↓
      LLM Layer
         ↓
 Human Validation
         ↓
 Results Storage
         ↓
 FastAPI
         ↓
 Dashboard / Reporting
```

## Build

Use:

- Python
- FastAPI
- Docker
- PyTest
- SentenceTransformers
- BERTopic
- scikit-learn
- Streamlit

## Suggested API Endpoints

```text
POST /feedback/analyse

GET /themes

GET /themes/{theme_id}

GET /themes/{theme_id}/examples

POST /themes/{theme_id}/approve

POST /themes/{theme_id}/merge

GET /model/metrics
```

---

## Human-in-the-Loop Workflow

If confidence is low:

```text
New theme detected
       ↓
confidence < threshold
       ↓
Send for analyst review
       ↓
Accept / Merge / Rename / Reject
```

## Production Monitoring

Track:

```text
number of comments
average comment length
theme distribution
unknown-theme rate
low-confidence rate
new-theme rate
embedding drift
LLM failure rate
LLM latency
LLM cost
```

Example review rule:

```text
IF low-confidence rate > 20%
OR unknown-theme rate > 10%
OR major embedding drift detected
THEN trigger analyst review
```

## Deliverables

```text
api/
└── main.py

app/
└── streamlit_app.py

tests/

Dockerfile

architecture.md
```

## Key Learning Outcome

Be able to explain:

> How would this system move from an experiment into a maintained university data product?

---

# Day 8 — Portfolio, Documentation and Interview Preparation

## Goal

Convert the technical work into a polished portfolio project and interview asset.

## Repository Structure

```text
student-voice-intelligence/

├── README.md
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_embeddings.ipynb
│   ├── 04_bertopic.ipynb
│   └── 05_llm_analysis.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── embeddings/
│   ├── clustering/
│   ├── thematic_analysis/
│   ├── llm/
│   └── evaluation/
│
├── evaluation/
│   ├── gold_labels.csv
│   ├── test_cases.yaml
│   └── benchmark.py
│
├── api/
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── tests/
├── Dockerfile
├── requirements.txt
└── architecture.md
```

## README Story

Explain the project as:

### Problem

Universities receive large volumes of free-text student feedback that are difficult to analyse consistently.

### Objective

Determine whether NLP and LLM-assisted thematic analysis can produce useful, stable and reproducible themes while retaining analyst oversight.

### Methods Compared

```text
TF-IDF
→ Classical ML

Sentence Embeddings
→ PCA / UMAP
→ Clustering

BERTopic
→ Unsupervised theme discovery

LLM
→ Theme naming
→ Multi-label classification
→ Grounded summarisation
```

### Evaluation

Include:

- gold-labelled dataset
- classification metrics
- theme stability
- human agreement
- LLM reliability tests
- groundedness evaluation

### Production Design

Explain:

- FastAPI serving
- Dockerisation
- human review
- versioning
- monitoring
- continuous improvement

---

# Final Definition of Done

At the end of Day 8, the project should contain five major outputs.

## 1. Working Thematic Analysis Pipeline

Comparison of at least three approaches:

```text
TF-IDF
Embeddings / clustering
BERTopic / LLM-assisted analysis
```

## 2. Evaluation Benchmark

A manually reviewed gold dataset with actual metrics.

## 3. Human-in-the-Loop AI Workflow

Demonstrate that uncertain AI outputs can be reviewed, corrected or rejected.

## 4. Production-Style Application

At minimum:

```text
FastAPI
Docker
PyTest
Streamlit
basic monitoring
```

## 5. Interview-Ready Documentation

You should be able to clearly explain:

- why each method was selected
- alternatives considered
- model limitations
- evaluation methodology
- failure cases
- responsible AI considerations
- how the solution would operate in production
- how the system would be continuously improved

---

# Final CV Project Description

Once completed, the project can appear on the CV as:

> **LLM-Assisted Student Feedback Thematic Analysis** — Developed an NLP and GenAI pipeline for extracting actionable themes from unstructured student feedback, comparing TF-IDF, semantic embeddings, BERTopic and LLM-assisted thematic coding. Built a human-labelled evaluation benchmark measuring multi-label classification, theme stability and LLM reliability, and implemented human-in-the-loop review, FastAPI serving, Docker deployment and monitoring for production-style delivery.

---

# Core Skill Gaps This Project Is Designed to Close

```text
Thematic analysis
        ✓

NLP
        ✓

Sentence transformers / embeddings
        ✓

Dimensionality reduction
        ✓

Clustering
        ✓

BERTopic
        ✓

LLM evaluation
        ✓

Method selection and justification
        ✓

Human-in-the-loop AI
        ✓

Production architecture
        ✓

FastAPI
        ✓

Docker
        ✓

Testing
        ✓

Monitoring
        ✓

Continuous improvement
        ✓

Responsible AI thinking
        ✓
```

The project is therefore not just a portfolio exercise. Its purpose is to give you concrete evidence for the exact areas where the UNSW Senior Data Scientist role currently stretches your profile.