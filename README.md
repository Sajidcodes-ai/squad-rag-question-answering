# Wikipedia Question Answering with RAG (SQuAD, FAISS and Flan-T5)

A Retrieval-Augmented Generation (RAG) system that answers questions about Wikipedia articles. The system first retrieves the most relevant paragraphs with semantic search, then a language model reads them and writes a short answer. The pipeline is built and evaluated on the real-world SQuAD v1.1 dataset.

## Overview

A normal language model answers from memory. A RAG system works like an open-book exam: it looks up the relevant text first, then answers using that text. This makes answers easier to check, because every answer can be traced back to a source paragraph.

**Pipeline**

1. **Prepare** the data: reshape the raw file into a paragraph table (knowledge base) and a question table (with ground-truth answers).
2. **Embed** every paragraph with a sentence-transformer model.
3. **Index** the embeddings in a FAISS vector store.
4. **Retrieve** the top-3 most similar paragraphs for a question.
5. **Generate** a short answer from the question and the retrieved paragraphs.
6. **Evaluate** retrieval quality (Recall@k) and answer quality (Exact Match and F1).

## Dataset

- **SQuAD v1.1 (development set)**: Stanford Question Answering Dataset, built from Wikipedia articles.
- Official source: https://rajpurkar.github.io/SQuAD-explorer/
- Size used in this project: **48 articles, 2,067 paragraphs, 10,570 questions**. Most questions have several accepted answers.
- The raw file is a flattened (wide) CSV with 48 rows and 9,825 columns, so the notebook first reshapes it into two clean tables.

## Tech Stack

| Component | Tool |
|---|---|
| Language and environment | Python, Google Colab (free T4 GPU) |
| Data handling and charts | pandas, numpy, matplotlib, seaborn |
| Embeddings | `sentence-transformers` with `all-MiniLM-L6-v2` (384 dimensions) |
| Vector search | FAISS (`IndexFlatIP`, normalized vectors, so inner product equals cosine similarity) |
| Generator | `google/flan-t5-base` (Hugging Face Transformers, PyTorch) |

## Results

**Retrieval** (all 10,570 questions): how often the correct source paragraph appears in the top k results.

| Metric | Score |
|---|---|
| Recall@1 | 62.51% |
| Recall@3 | 80.77% |
| Recall@5 | 86.40% |
| Recall@10 | 91.72% |

**Full RAG pipeline** (random sample of 500 questions, top-3 paragraphs given to the generator):

| Metric | Score |
|---|---|
| Exact Match | 65.40% |
| F1 score | 74.22% |

Exact Match counts an answer as correct only if it matches one of the accepted answers after normalization (lowercase, no punctuation, no articles). F1 gives partial credit for overlapping words. For each question, the best score over all accepted answers is used.

## Key Observations

- Retrieval is a limiting factor: for about 19% of questions, the source paragraph is not in the top 3, so the generator does not see it.
- Wikipedia articles contain many similar paragraphs (for example, many paragraphs about the same event), which explains why Recall@1 is lower than Recall@3.
- The retrieval metric is strict, because it only counts the exact paragraph the question was written from. Another paragraph may also contain the answer, so real usefulness can be somewhat higher.
- The gap between Exact Match (65.40%) and F1 (74.22%) shows that many answers are partly correct or worded differently from the reference.

## Limitations

- The end-to-end evaluation uses a random sample of **500 questions** (fixed with `random_state=42`), not the full 10,570, to keep generation time reasonable.
- `flan-t5-base` was trained on a large mix of question-answering data, which may include SQuAD-style data. Scores may therefore be somewhat higher than on completely unseen data.
- The knowledge base contains only the 48 Wikipedia articles of the SQuAD development set. Questions about other topics will not be answered reliably.

## Ideas for Improvement

- Try a larger embedding model or add a re-ranking step to improve Recall@1.
- Combine semantic search with keyword search (hybrid retrieval).
- Try a larger generator model.
- Evaluate on all 10,570 questions.

## How to Run

1. Open the notebook `squad_rag_question_answering.ipynb` in Google Colab.
2. Set the runtime to GPU: **Runtime > Change runtime type > T4 GPU**.
3. Upload `dev-v1_1.csv` to the Colab files panel (the notebook reads it from `/content/dev-v1_1.csv`).
4. Run all cells from top to bottom. The notebook installs `sentence-transformers` and `faiss-cpu` in its first cell.

To run locally instead, install the libraries in `requirements.txt` and adjust the file path in the data-loading cell.

## Repository Contents

- `squad_rag_question_answering.ipynb`: the full notebook with real outputs (EDA, visualization, embeddings, FAISS index, retriever, generator, evaluation, custom question cell)
- `requirements.txt`: Python libraries used
- `README.md`: this file

## Author

**Sajid Hussain**: https://github.com/Sajidcodes-ai
