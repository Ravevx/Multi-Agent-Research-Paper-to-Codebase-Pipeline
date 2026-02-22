# Multi-Agent Research Paper to Codebase Pipeline

[![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-orange.svg)](https://crewai.com)
[![FAISS](https://img.shields.io/badge/FAISS-RAG%20Vector%20Search-purple.svg)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-brightgreen.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Upload any research paper (PDF or arXiv ID) → AI agents analyze it → Generate a complete, runnable Python codebase → Download as ZIP**

Transform arXiv papers into production-ready projects using a multi-agent workflow
**Analyst → Architect → Coder**.

---

##Table of Contents

- [What It Does](#-what-it-does)
- [Generated Project Example](#-generated-project-example)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Usage Flow](#-usage-flow)
- [Configuration](#-configuration)
- [Development & Debugging](#-development--debugging)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## What It Does

Most research papers are never implemented. Reading a paper, understanding its architecture, designing the project structure, and writing the code can take days — even for experienced engineers.

This pipeline does it automatically:

1. **Reads** the paper using RAG (FAISS vector search over paper chunks)
2. **Analyses** it, extracts components, equations, data flow, inputs/outputs
3. **Plans** a full Python project, files, classes, functions, algorithm steps
4. **Codes** every file — using the most relevant paper sections as context per file
5. **Packages** everything — folder structure, `requirements.txt`, `README.md`, downloadable ZIP

---
## Generated Project Example

For **"Attention Is All You Need"** paper:

```
output/projects/transformer-attention/
├── src/
│   ├── encoder.py         # MultiHeadAttention + FeedForward
│   ├── decoder.py         # DecoderLayer with encoder-decoder attention
│   ├── attention.py       # ScaledDotProductAttention
│   ├── model.py           # Transformer
│   └── main.py            # End-to-end demo
├── requirements.txt       # torch, numpy
└── README.md              # How to train/extend
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Paper Ingestion** | Upload local PDF or enter arXiv ID automatic text extraction and chunking |
| **FAISS RAG** | Vector index over paper chunks per-agent, per-file context retrieval |
| **Multi-Agent Pipeline** | Analyst → Architect → Coder three specialized agents, each with targeted prompts |
| **Streamlit UI** | Step-by-step flow: upload → review plan → generate code → download ZIP |
| **Plan Review** | Inspect every file, class, function, and algorithm step before generating code |
| **Project Export** | Full folder structure with `src/`, `requirements.txt`, `README.md` per project |
| **Syntax Validation** | `ast.parse()` on every generated file, broken files flagged in UI |
| **Post-Processing** | Auto-fix `main_file` prefix, auto-collect dependencies, phantom class removal |
| **Debug Logs** | Raw agent outputs saved per run for prompt tuning and debugging |

---
## 🚀 Quick Start

### 1. Clone \& Install

```bash
git clone https://github.com/Ravevx/multi-agent-paper-pipeline.git
cd multi-agent-paper-pipeline
```

```bash
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```


### 2. Configure LLM

**Local LM Studio (default):**

```bash
# Start LM Studio server at http://127.0.0.1:1234/v1
# Set your model in config.py
LMSTUDIO_URL=http://127.0.0.1:1234/v1
LMSTUDIO_MODEL=your-model-name
```

### 3. Run

```bash
streamlit run app.py
```

## Usage Flow

```
1. Upload PDF or enter arXiv ID
2. [Analyse] → Build RAG → Analyst → Architect → Plan
3. Review plan (files, classes, functions, steps)
4. [Approve] → Generate code file-by-file
5. Download ZIP or use local copy
```

### Agent Roles

| Agent | Input | Output | Key Prompt Rule |
|-------|-------|--------|-----------------|
| **Analyst** | Paper chunks (RAG) | 5-section technical analysis | Start with `## 1.` no preamble, no hedging |
| **Architect** | Analysis text | ProjectPlan JSON | Class names must cross-reference, no phantom classes |
| **Coder** | File spec + RAG context | Python source file | EXACT NAMES box at top, enforces plan class names |

### Key Design Decisions

- **RAG per file**: Each file gets its own targeted RAG query using its class names and logic summary, so `attention.py` retrieves attention equation chunks, not encoder chunks
- **Plan validation**: `_validate_plan()` removes phantom classes (classes listed in `main.py` but not defined anywhere), preventing `ImportError` at runtime
- **nn.Module enforcement**: `fix_nn_module()` detects torch files and adds `(torch.nn.Module)` inheritance where missing
- **No architect truncation fallback**: If `main.py` algorithm steps are empty (truncated output), `crew_runner.py` rebuilds them from the actual class/function list

---


## Project Structure

```
.
├── app.py                 # Streamlit app
├── crew_runner.py         # Main orchestration
├── crew_tasks.py          # Agent prompts
├── crew_agents.py         # Agent definitions
├── rag_store.py           # FAISS RAG
├── paper_tools.py         # PDF/arXiv extraction
├── project_planner.py     # Data models
├── config.py              # Settings
├── requirements.txt       # Dependencies
├── output/
│   ├── papers/            # Cached PDFs
│   └── projects/          # Generated codebases
│       └── <project_name>/
│           ├── src/
│           ├── requirements.txt
│           └── README.md
└── screenshots/           # UI screenshots
```

##Performance

| Paper | Files | Time | Syntax Errors |
| :-- | :-- | :-- | :-- |
| Attention Is All You Need | 7 | 12 min | 0 |
| BERT | 12 | 18 min | 1 |
| GPT-2 | 9 | 14 min | 0 |


## License

MIT License  [LICENSE](LICENSE)

##Acknowledgments

- [CrewAI](https://crewai.com) — multi-agent orchestration
- [FAISS](https://github.com/facebookresearch/faiss) — vector search
- [Streamlit](https://streamlit.io) — amazing UI framework
- Research community — for the papers that power this!
***


