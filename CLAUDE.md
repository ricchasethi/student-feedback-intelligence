# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Planning stage, with no code yet. `student-voice-intelligence.md` is the source of truth: an 8-day plan for an LLM-assisted thematic analysis system for student feedback. Read the relevant "Day N" section before building any part of it, and follow its deliverables and target repo layout (Day 8, "Repository Structure") rather than inventing a different structure.

Core question the project must answer: can NLP and LLM-assisted thematic analysis turn large volumes of student feedback into stable, interpretable, actionable themes while keeping human oversight? The project has to show *why* each method was chosen, how it was evaluated, and where it fails, not only that it runs.

## Dataset

`load_dataset("Hannes111/student_feedback_sentiment")` from Hugging Face: one `train` split with 724 rows and columns `text, category, sentiment_score, sentiment`. It has 6 categories and positive/neutral/negative sentiment.

## Working mode

The user is learning through this project. Claude acts as the teacher, giving one lesson per plan day: concepts, then tasks, then review. The user writes the notebook code. Review it and give hints, and only write full solutions when asked.

## Planned shape

Progression (each day builds on the last; earlier methods are baselines for later ones):
1. EDA and data quality on a public student-feedback dataset, including a PII check and a multi-theme check. Save a cleaned dataset.
2. Baselines: TF-IDF → LogReg for category, TF-IDF → LogReg for sentiment, TF-IDF → KMeans.
3. SentenceTransformer embeddings and clustering.
4. BERTopic and theme taxonomy design.
5. LLM layer, used only for controlled tasks: theme naming from cluster keywords and examples, multi-label theme × sentiment classification, and grounded summaries.
6. Evaluation: gold dataset, classification metrics, clustering stability, an LLM reliability test suite, and groundedness.
7. Production: FastAPI (`api/main.py`), Streamlit (`app/streamlit_app.py`), Docker, pytest, a human-in-the-loop review flow and monitoring.

Planned layout: `notebooks/01_eda … 05_llm_analysis`, `src/{preprocessing,embeddings,clustering,thematic_analysis,llm,evaluation}/`, `evaluation/{gold_labels.csv,test_cases.yaml,benchmark.py}`, `api/`, `app/`, `tests/`, `requirements.txt`, `architecture.md`.

## Rules the plan sets for the LLM layer

- Never send the whole dataset to an LLM. Use it only where it adds measurable value over the non-LLM baselines.
- Force structured (JSON) outputs and handle invalid responses.
- Every summary stores `supporting_comment_ids` so it can be audited against real comments.
- Record the prompt version, model version, latency and cost for each call.
- Low-confidence or new themes go to analyst review (accept, merge, rename or reject) rather than being auto-accepted.

## Environment and commands

uv, Python 3.12 (`.python-version`). Dependencies live in `pyproject.toml` and `uv.lock`.

```bash
uv sync                          # create/refresh .venv from the lock (includes dev group)
uv add <pkg>  /  uv add --dev <pkg>
uv run pytest                    # tests/ (none yet)
uv run pytest tests/test_x.py::test_name
uv run ruff check .
uv run jupyter lab               # notebooks/
uv run hf auth login             # Hugging Face CLI ships with huggingface-hub (no [cli] extra in 1.x+)
```

Dependency gotchas:
- `torch` comes from the PyTorch **CPU-only** index (`[tool.uv.sources]`) because this machine has no NVIDIA GPU. Remove that source to get CUDA wheels.
- Keep the lower bounds on `sentence-transformers` (`>=5`) and `datasets` (`>=4`). Without them uv backtracks to 2020–21 releases: `huggingface-hub` 2.x is newer than any `transformers` supports, and that pulls in an old `tokenizers` that needs a Rust compiler. Old `datasets` also crashes on import with current `pyarrow`.
- If uv warns that `VIRTUAL_ENV` doesn't match `.venv`, another project's venv is active in the shell. Run `deactivate` or ignore the warning (uv uses `.venv`).
