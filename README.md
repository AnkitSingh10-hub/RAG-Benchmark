<img width="1848" height="873" alt="image" src="https://github.com/user-attachments/assets/e2742b91-4ec5-4b1d-b754-bec1c7dec84b" />
<img width="1855" height="646" alt="image" src="https://github.com/user-attachments/assets/5fca9ecc-44a3-4762-a7fe-1167a9429031" />
# Insurellm RAG — Retrieval-Augmented Generation System

This project implements and evaluates a Retrieval-Augmented Generation (RAG)
chatbot for a fictional insurance company, Insurellm. Two independent
implementations were built over the same knowledge base so that different
retrieval and generation design choices could be compared empirically rather
than assumed.

## Evaluation Methodology

Both implementations are evaluated against the same held-out set of
140–150 test questions, spanning multiple reasoning categories (direct
fact lookup, temporal, comparative, numerical, relational, multi-hop/spanning,
and holistic questions requiring aggregation across documents).

Two families of metrics are measured for every run:

**Retrieval quality**
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG)
- Precision@k
- Recall@k
- Keyword coverage

**Generated answer quality** (scored by an LLM-as-judge against a reference answer, 1–5 scale)
- Accuracy
- Completeness
- Relevance

Each evaluation run is exported to a timestamped Excel file in `eval_results/`,
recording the run configuration alongside the full metric breakdown, so
different design choices can be compared across runs.

## R&D Techniques Investigated

Over the course of this project the following RAG improvement techniques
were implemented and evaluated:

1. **Chunking strategy** — multiple chunking approaches were implemented and
   compared (doc-type-aware heading splitting, plain heading splitting,
   character-based splitting, recursive character splitting).
2. **Encoder (embedding model) selection** — several embedding models were
   evaluated against the test set to select the best-performing encoder.
3. **Prompt improvements** — system prompts were refined to include general
   company context, the current date, and relevant retrieved context and
   conversation history.
4. **LLM-based document pre-processing** — an LLM was used to generate chunks
   directly from source documents (headline, summary, and original text),
   rather than relying purely on rule-based splitting.
5. **Query rewriting** — an LLM rewrites the user's raw question into a more
   targeted retrieval query before the vector search step.
6. **Re-ranking** — an LLM reranks a wider initial retrieval set before the
   top results are passed into the generation prompt.

Query expansion (turning one question into multiple retrieval queries) was
scoped as a technique but is not yet implemented in either pipeline.

## Project Structure

```
.
├── knowledge_base/          # shared source documents
├── implementation/          # LangChain-based RAG pipeline
├── pro_implementation/      # custom, LangChain-free RAG pipeline
├── vanilla_evaluation/      # evaluation harness for implementation/
├── evaluation/              # evaluation harness for pro_implementation/
├── vanilla_evaluator.py     # evaluation dashboard for implementation/
├── evaluator.py             # evaluation dashboard for pro_implementation/
└── eval_results/            # exported Excel logs of every evaluation run
```

## Setup

```bash
uv sync
```

Create a `.env` file in the project root with the required API credentials
(Azure AI Foundry API key, and a Mistral API key if using LLM-based
chunking).

## Running the Pipelines

**LangChain-based implementation:**

```bash
uv run -m implementation.ingest      # build the vector store
uv run vanilla_evaluator.py          # launch the evaluation dashboard
```

**Custom implementation:**

```bash
uv run -m pro_implementation.ingest  # build the vector store
uv run evaluator.py                  # launch the evaluation dashboard
```

Each dashboard runs the retrieval and answer evaluations described above and
allows the results to be exported to `eval_results/` as an Excel file.

