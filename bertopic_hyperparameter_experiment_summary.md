# BERTopic Hyperparameter Experiment Summary

Updated: 2026-05-22

This note summarizes the experiment logic and current results for the three BERTopic corpus-clustering experiments:

- HotpotQA Wikipedia abstracts
- MultiHop-RAG corpus
- Open-NQ / DPR Wikipedia passages

The goal of these experiments is to find a BERTopic configuration that gives high topic coherence, acceptable topic diversity, reasonable outlier ratio, and stable natural topic counts. Topic number is not fixed manually in these searches. BERTopic uses `nr_topics=None`, so UMAP + HDBSCAN decide the natural number of topics.

## Common Pipeline

All three experiments follow the same conceptual BERTopic flow:

1. Read corpus documents.
2. Clean and preprocess text.
3. Encode each document with `all-MiniLM-L6-v2`.
4. Run BERTopic with precomputed embeddings.
5. Let UMAP reduce embedding dimensionality.
6. Let HDBSCAN cluster documents.
7. Use c-TF-IDF / CountVectorizer to extract topic words.
8. Evaluate topics with:
   - `TC(NPMI)`
   - `TC(Cv)`
   - `TC mean = (NPMI + Cv) / 2`
   - `TD`
   - outlier ratio
   - actual topic count
   - topic size distribution

Shared important settings:

| Setting | Value |
|---|---:|
| Embedding model | `all-MiniLM-L6-v2` |
| BERTopic topic mode | `nr_topics=None` |
| Vectorizer max features | `10000` |
| UMAP min dist | `0.0` |
| HDBSCAN selection method | `eom` |
| BERTopic probabilities | `False` |
| OCTIS top-k | `10` |

## Experiment Designs

### HotpotQA

Script:

`hotpotqa/run_bertopic_hyperparameter_search.py`

Corpus:

`hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2`

Current design:

- Large corpus, so the experiment uses staged random sampling.
- A sampling pool of `200,000` documents is prepared by default.
- Each sample draws documents from this cached pool and reuses the aligned embeddings.

Search grid:

| Parameter | Values |
|---|---|
| `UMAP_N_COMPONENTS` | `[5, 10]` |
| `UMAP_N_NEIGHBORS` | `[5, 10, 15, 30]` |
| `HDBSCAN_MIN_CLUSTER_SIZE` | `[10, 20, 50, 100, 200]` |
| `HDBSCAN_MIN_SAMPLES` | `[5, 10, 20]` |
| `MAX_DF` | `[0.85, 0.95, 1.0]` |
| `NGRAM_RANGE` | `[(1, 1), (1, 2)]` |
| Vectorizer `min_df` | `2` |

Sampling stages:

| Stage | Candidates | Samples per candidate | Sample size |
|---:|---:|---:|---:|
| 1 | 720 | 5 | 10,000 |
| 2 | top 30 from stage 1 | 50 | 10,000 |
| 3 | top 5 from stage 2 | 30 | 50,000 |

Total training samples:

`720 * 5 + 30 * 50 + 5 * 30 = 5250`

Ranking rule:

```text
ranking_score = mean(TC)
              - 0.5 * std(TC)
              + 0.05 * mean(TD)
              - 0.10 * mean(outlier_ratio)
              - 0.10 * topic_count_cv
```

Current result source:

`hotpotqa/result/bertopic_hyperparameter_search/20260521_234204`

Completion status:

| Metric | Value |
|---|---:|
| Total sample rows | 5,250 |
| OK rows | 5,201 |
| Error rows | 49 |
| Final stage candidates | 5 |
| Samples per final candidate | 30 |

Best final configuration:

| Field | Value |
|---|---:|
| Iteration | `304` |
| Run label | `iter_0304_uc5_un30_hcs20_hms20_df0p95_ng1_2` |
| UMAP components | `5` |
| UMAP neighbors | `30` |
| HDBSCAN min cluster size | `20` |
| HDBSCAN min samples | `20` |
| max_df | `0.95` |
| ngram range | `(1, 2)` |
| Mean TC | `0.5915` |
| Std TC | `0.0059` |
| Mean TC(NPMI) | `0.4091` |
| Mean TC(Cv) | `0.7739` |
| Mean TD | `0.7691` |
| Mean outlier ratio | `0.4223` |
| Mean topic count | `251.6` |
| Topic count std | `9.08` |
| Ranking score | `0.5812` |

Recommended topic count for later experiments:

Approximately `252` natural topics, based on the best final candidate's mean topic count.

Main output files:

- `sampling_partial_results.csv`
- `sampling_stage_1_aggregate.csv`
- `sampling_stage_2_aggregate.csv`
- `sampling_stage_3_aggregate.csv`
- `sampling_final_ranking.csv`
- `sampling_best_hyperparameters.json`
- `sampling_hyperparameter_search_report.md`
- `sampling_final_tc_mean_vs_std.png`
- `sampling_final_top_ranking_score.png`
- `sampling_final_topic_count_stability.png`
- `sampling_final_td_vs_tc.png`
- `sampling_final_outlier_vs_tc.png`

### MultiHop-RAG

Script:

`multihop_qa/run_bertopic_hyperparameter_search.py`

Corpus:

`multihop_qa/corpus.json`

Document fields:

- `title` is kept as metadata.
- `body` is the main text used for topic clustering.

Current design:

- Small corpus, so the experiment runs the full grid directly.
- No sampling stages are used.
- The script is pinned to the second GPU, `cuda:1`.

Search grid:

| Parameter | Values |
|---|---|
| `UMAP_N_COMPONENTS` | `[5, 10]` |
| `UMAP_N_NEIGHBORS` | `[5, 10, 15, 30]` |
| `HDBSCAN_MIN_CLUSTER_SIZE` | `[5, 10, 20, 30, 50]` |
| `HDBSCAN_MIN_SAMPLES` | `[1, 3, 5]` |
| `MAX_DF` | `[0.85, 0.95, 1.0]` |
| `NGRAM_RANGE` | `[(1, 1), (1, 2)]` |
| Vectorizer `min_df` | `1` |

Total grid size:

`2 * 4 * 5 * 3 * 3 * 2 = 720`

Current result source:

`multihop_qa/multi_hop_result/bertopic_hyperparameter_search/20260520_181754`

Completion status:

| Metric | Value |
|---|---:|
| Total grid rows | 720 |
| OK rows | 720 |
| Error rows | 0 |

Best configuration by `TC mean`:

| Field | Value |
|---|---:|
| Iteration | `368` |
| Run label | `iter_0368_uc10_un5_hcs5_hms3_df0p85_ng1_2` |
| UMAP components | `10` |
| UMAP neighbors | `5` |
| HDBSCAN min cluster size | `5` |
| HDBSCAN min samples | `3` |
| max_df | `0.85` |
| ngram range | `(1, 2)` |
| TC mean | `0.6926` |
| TC(NPMI) | `0.5214` |
| TC(Cv) | `0.8638` |
| TD | `0.9240` |
| Outlier ratio | `0.0772` |
| Actual topic count | `50` |

Recommended topic count for later experiments:

`50` natural topics.

Main output files:

- `partial_hyperparameter_search_results.csv`
- `hyperparameter_search_results.csv`
- `hyperparameter_search_report.md`
- `search_grid.json`
- `hyperparameter_metric_comparison.png`
- `outlier_ratio_by_iteration.png`
- `actual_topic_count_by_iteration.png`

### Open-NQ / DPR

Script:

`nq/run_bertopic_hyperparameter_search.py`

Corpus:

`facebook/wiki_dpr`, config `psgs_w100.nq.compressed`

The script reads DPR parquet shards directly and keeps only:

- `id`
- `title`
- `text`

The DPR-provided embedding column is ignored because BERTopic uses the configured sentence-transformer embeddings.

Current design:

- Large corpus, so the experiment uses the same staged sampling design as HotpotQA.
- Default sampling pool is `200,000` DPR passages.
- Embedding and cuML modeling are pinned to the second GPU, `cuda:1`.

Search grid:

| Parameter | Values |
|---|---|
| `UMAP_N_COMPONENTS` | `[5, 10]` |
| `UMAP_N_NEIGHBORS` | `[5, 10, 15, 30]` |
| `HDBSCAN_MIN_CLUSTER_SIZE` | `[10, 20, 50, 100, 200]` |
| `HDBSCAN_MIN_SAMPLES` | `[5, 10, 20]` |
| `MAX_DF` | `[0.85, 0.95, 1.0]` |
| `NGRAM_RANGE` | `[(1, 1), (1, 2)]` |
| Vectorizer `min_df` | `2` |

Sampling stages:

| Stage | Candidates | Samples per candidate | Sample size |
|---:|---:|---:|---:|
| 1 | 720 | 5 | 10,000 |
| 2 | top 30 from stage 1 | 50 | 10,000 |
| 3 | top 5 from stage 2 | 30 | 50,000 |

Total training samples:

`720 * 5 + 30 * 50 + 5 * 30 = 5250`

Ranking rule:

```text
ranking_score = mean(TC)
              - 0.5 * std(TC)
              + 0.05 * mean(TD)
              - 0.10 * mean(outlier_ratio)
              - 0.10 * topic_count_cv
```

Current result source:

`nq/nq_result/bertopic_hyperparameter_search/20260521_234931`

Completion status:

| Metric | Value |
|---|---:|
| Total sample rows | 5,250 |
| OK rows | 5,070 |
| Error rows | 180 |
| Final stage candidates | 5 |
| Samples per final candidate | 30 |

Best final configuration:

| Field | Value |
|---|---:|
| Iteration | `302` |
| Run label | `iter_0302_uc5_un30_hcs20_hms20_df0p85_ng1_2` |
| UMAP components | `5` |
| UMAP neighbors | `30` |
| HDBSCAN min cluster size | `20` |
| HDBSCAN min samples | `20` |
| max_df | `0.85` |
| ngram range | `(1, 2)` |
| Mean TC | `0.4901` |
| Std TC | `0.0054` |
| Mean TC(NPMI) | `0.2952` |
| Mean TC(Cv) | `0.6850` |
| Mean TD | `0.7959` |
| Mean outlier ratio | `0.5154` |
| Mean topic count | `200.3` |
| Topic count std | `10.37` |
| Ranking score | `0.4704` |

Recommended topic count for later experiments:

Approximately `200` natural topics, based on the best final candidate's mean topic count.

Main output files:

- `sampling_partial_results.csv`
- `sampling_stage_1_aggregate.csv`
- `sampling_stage_2_aggregate.csv`
- `sampling_stage_3_aggregate.csv`
- `sampling_final_ranking.csv`
- `sampling_best_hyperparameters.json`
- `sampling_hyperparameter_search_report.md`
- `sampling_final_tc_mean_vs_std.png`
- `sampling_final_top_ranking_score.png`
- `sampling_final_topic_count_stability.png`
- `sampling_final_td_vs_tc.png`
- `sampling_final_outlier_vs_tc.png`

## Cross-Dataset Comparison

| Dataset | Search type | Best UMAP | Best HDBSCAN | Best max_df | Best ngram | TC mean | TD | Outlier ratio | Topic count |
|---|---|---|---|---:|---|---:|---:|---:|---:|
| HotpotQA | staged sampling | components=5, neighbors=30 | min_cluster=20, min_samples=20 | 0.95 | (1, 2) | 0.5915 | 0.7691 | 0.4223 | 251.6 |
| MultiHop-RAG | full grid | components=10, neighbors=5 | min_cluster=5, min_samples=3 | 0.85 | (1, 2) | 0.6926 | 0.9240 | 0.0772 | 50 |
| Open-NQ/DPR | staged sampling | components=5, neighbors=30 | min_cluster=20, min_samples=20 | 0.85 | (1, 2) | 0.4901 | 0.7959 | 0.5154 | 200.3 |

Observations:

- `NGRAM_RANGE=(1, 2)` is selected for all three datasets, so bigrams are useful for the topic-word representation.
- MultiHop-RAG prefers much smaller HDBSCAN cluster sizes because the corpus is small.
- HotpotQA and NQ both converge to `UMAP_N_COMPONENTS=5`, `UMAP_N_NEIGHBORS=30`, `HDBSCAN_MIN_CLUSTER_SIZE=20`, and `HDBSCAN_MIN_SAMPLES=20`.
- Open-NQ has the highest outlier ratio among the three, which is expected because DPR passages are short and diverse.
- MultiHop-RAG has the cleanest clustering result in the current experiments, with high TD and low outlier ratio.

## How To Interpret The Metrics

`TC(NPMI)` and `TC(Cv)` measure topic coherence from different perspectives. In this project, `TC mean` is used as the main quality signal.

`TD` measures how diverse the top topic words are across topics. A high TD means topics are less likely to reuse the same words.

`outlier ratio` is the percentage of documents assigned to topic `-1` by HDBSCAN. A very high outlier ratio means many documents were not confidently clustered.

`actual topic count` is the number of non-outlier topics produced naturally by HDBSCAN. This is the topic number to use as a reference for later fixed-topic experiments.

`topic_count_cv` is used only in the staged sampling ranking. It penalizes configurations whose topic counts fluctuate strongly across random samples.

## Recommended Next Use

For later experiments, use the selected hyperparameters and natural topic counts as the first candidate settings:

| Dataset | Recommended hyperparameters | Recommended topic count |
|---|---|---:|
| HotpotQA | `uc=5`, `un=30`, `hcs=20`, `hms=20`, `max_df=0.95`, `ngram=(1,2)` | about 252 |
| MultiHop-RAG | `uc=10`, `un=5`, `hcs=5`, `hms=3`, `max_df=0.85`, `ngram=(1,2)` | 50 |
| Open-NQ/DPR | `uc=5`, `un=30`, `hcs=20`, `hms=20`, `max_df=0.85`, `ngram=(1,2)` | about 200 |

For HotpotQA and Open-NQ, the selected topic count comes from repeated random samples, not a full-corpus BERTopic fit. Before using the value as a final fixed topic number, it is reasonable to rerun the best configuration on a larger pool or the full corpus if compute budget allows.
