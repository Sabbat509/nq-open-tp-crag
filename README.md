# NQ-Open Topic-Partitioned CRAG

Topic-partitioned corrective RAG evaluation on the NQ-Open test set from
[`ehsk/OpenQA-eval`](https://github.com/ehsk/OpenQA-eval).

This repo contains only the TP-CRAG variant. It combines the CRAG process from
[`HuskyInSalt/CRAG`](https://github.com/HuskyInSalt/CRAG) with Kiraffe-style
topic partitioning from
[`Kiraffe1206/BERTopic-for-RAG`](https://github.com/Kiraffe1206/BERTopic-for-RAG).
The completed result below is the earlier fullwiki-backed run. Kiraffe's 2026-05-22 update recommends rebuilding NQ-Open on the DPR passage corpus (`facebook/wiki_dpr`, config `psgs_w100.nq.compressed`) with natural BERTopic topics instead of reusing the HotpotQA fullwiki partition.

## Current Result

Dataset: NQ-Open test set, 3,610 questions. This is the legacy fullwiki-backed result kept for traceability, not the updated DPR-aligned Kiraffe rerun.

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

## Kiraffe 2026-05-22 DPR Update

Kiraffe's updated NQ/Open-NQ recommendation is corpus-specific. The topic partition should be built from `facebook/wiki_dpr` using config `psgs_w100.nq.compressed`, not from the HotpotQA fullwiki abstracts index. The recommended setting is:

```text
uc=5, un=30, hcs=20, hms=20, max_df=0.85, ngram=(1,2), nr_topics=None
```

Kiraffe reports approximately 200 natural topics from repeated staged samples. A fully aligned rerun therefore needs these steps:

1. Build DPR document data from `facebook/wiki_dpr`, keeping only `id`, `title`, and `text`.
2. Build a DPR vector index with `all-MiniLM-L6-v2`.
3. Build the BERTopic partition with the settings above.
4. Attach topic IDs to the DPR chunks.
5. Run `evaluate_nq_open_crag_topic_rag.py` against the DPR topic index.

The old fullwiki shortcut should not be reported as the final updated NQ-Open TP-CRAG result.

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

## DPR-Aligned Reproduce Path

Prepare DPR document data:

```bash
python prepare_nq_dpr_doc_data.py \
  --output-file datasets/nq_open/intermediate/dpr_doc_data.pkl
```

Build the DPR vector index:

```bash
python build_reference_index.py \
  --doc-data datasets/nq_open/intermediate/dpr_doc_data.pkl \
  --output-dir datasets/nq_open/indexes/dpr/vector_index \
  --embedding-model all-MiniLM-L6-v2 \
  --batch-size 32 \
  --chunk-size-words 120 \
  --chunk-overlap-words 20 \
  --min-chunk-words 20 \
  --device cuda:0
```

Build the DPR topic partition with Kiraffe's updated setting:

```bash
python build_topic_partition.py \
  --doc-data datasets/nq_open/intermediate/dpr_doc_data.pkl \
  --output-dir datasets/nq_open/indexes/dpr/topic_partition_kiraffe_20260522 \
  --nr-topics none \
  --umap-n-components 5 \
  --umap-n-neighbors 30 \
  --min-topic-size 20 \
  --hdbscan-min-samples 20 \
  --max-df 0.85 \
  --ngram-range 1,2 \
  --min-df 2 \
  --backend auto \
  --device cuda:0
```

Attach topics and evaluate:

```bash
python build_crag_topic_index.py \
  --base-index-dir datasets/nq_open/indexes/dpr/vector_index \
  --topic-map datasets/nq_open/indexes/dpr/topic_partition_kiraffe_20260522/topic_article_map.jsonl \
  --topic-info datasets/nq_open/indexes/dpr/topic_partition_kiraffe_20260522/topic_info_none.csv \
  --output-dir datasets/nq_open/indexes/dpr/topic_index_kiraffe_20260522

python evaluate_nq_open_crag_topic_rag.py \
  --data-file raw/nq_open_test.jsonl \
  --index-dir datasets/nq_open/indexes/dpr/topic_index_kiraffe_20260522 \
  --topic-info datasets/nq_open/indexes/dpr/topic_partition_kiraffe_20260522/topic_info_none.csv \
  --output-dir results_dpr_kiraffe_20260522 \
  --model gemma2:2b \
  --resume
```
