# NQ-Open Topic-Partitioned CRAG

Topic-partitioned corrective RAG evaluation on the NQ-Open test set from
[`ehsk/OpenQA-eval`](https://github.com/ehsk/OpenQA-eval).

This repo contains only the TP-CRAG variant. It combines the CRAG process from
[`HuskyInSalt/CRAG`](https://github.com/HuskyInSalt/CRAG) with Kiraffe-style
topic partitioning from
[`Kiraffe1206/BERTopic-for-RAG`](https://github.com/Kiraffe1206/BERTopic-for-RAG).
The run uses the prebuilt 40-topic full Wikipedia partition, routes each
question to likely topics, evaluates retrieval confidence, and applies
corrective fallback retrieval when evidence is weak.

## Result

Dataset: NQ-Open test set, 3,610 questions.

| Method | EM | Acc | F1 |
|---|---:|---:|---:|
| TP-CRAG | 22.55 | 31.55 | 29.90 |

LaTeX row:

```latex
TP-CRAG & 22.55 & 31.55 & 29.90
```

CRAG branch counts:

```text
correct: 940
ambiguous: 1136
incorrect: 1534
```

## Files

- `evaluate_nq_open_crag_topic_rag.py`: topic-partitioned CRAG evaluation script.
- `crag_topic_rag/`: CRAG retrieval evaluation and topic-routing helpers.
- `naive_rag/`: index and LLM client helpers.
- `raw/nq_open_test.jsonl`: NQ-Open questions and gold answer aliases extracted
  from `OpenQA-eval/data/model_outputs/NQ_FiD.jsonl`.
- `results/summary.json`: final metrics.
- `results/predictions.csv`: generated predictions.
- `results/records.jsonl`: full per-example records.

The full Wikipedia topic index and topic partition are intentionally not
committed because the underlying chunk index is several GB. Place or symlink
them at:

```text
datasets/nq_open/indexes/fullwiki/topic_index/
datasets/nq_open/indexes/fullwiki/topic_partition/
```

Expected topic files include:

```text
topic_index/manifest.json
topic_index/chunks.jsonl
topic_index/embeddings.npy
topic_partition/topic_info_40.csv
```

## Reproduce

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
ollama serve
ollama pull gemma2:2b

python evaluate_nq_open_crag_topic_rag.py \
  --data-file raw/nq_open_test.jsonl \
  --index-dir datasets/nq_open/indexes/fullwiki/topic_index \
  --topic-info datasets/nq_open/indexes/fullwiki/topic_partition/topic_info_40.csv \
  --output-dir results \
  --model gemma2:2b \
  --resume
```

## Metrics

- `EM`: normalized exact match against any gold answer alias.
- `Acc`: relaxed lexical accuracy: exact match or containment after
  normalization.
- `F1`: max token-level F1 over all gold answer aliases.
