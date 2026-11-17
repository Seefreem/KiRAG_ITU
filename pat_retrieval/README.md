# PAT retrieval module

Independent FAISS retrieval, PAT_BERT reranking, corpus loading, and per-beam
caching. No imports from KiRAG or `scripts/`. Install elsewhere with
`pip install ./pat_retrieval`, or copy the package into another project.

```python
from pat_retrieval import PATRetrievalPipeline, RetrievalConfig

pipeline = PATRetrievalPipeline('pat_retrieval/pat_retrieval_config.json')
# Alternatively: PATRetrievalPipeline(RetrievalConfig(...))
pipeline.reset()  # once per new question
results = pipeline(['Who ...?'])
next_results = pipeline(['Who ...? Additional reasoning context'])
```

The repository's `pat_retrieval/pat_retrieval_config.json` is the working example. All data
paths in a JSON config resolve relative to that config file. When constructing
`RetrievalConfig` directly, relative paths resolve from the working directory.
Encoder model/tokenizer names are Hugging Face identifiers or local paths;
local encoder paths resolve from the working directory.

| Setting | Meaning |
| --- | --- |
| `index_file` | FAISS index filename |
| `index_meta_file` | Pickled FAISS-row-to-document-ID sequence filename |
| `corpus_files` | One or more JSON array or `.jsonl` document files |
| `return_top_k` | Number of reranked documents returned, 1–508 |
| `cache_size` | Documents retained per beam, 0–507; default 58 |
| `beams` | Maximum query streams per call; default 1 |
| `return_items` | Any nonempty selection of `id`, `score`, `contents`, `embedding` |
| `pat_bert.checkpoint` | Trained checkpoint file; example preserves the requested path |
| `pat_bert.device` | Device for PAT_BERT inference |
| `encoder` | text embedder: Retrieval model, tokenizer, retriever type, device, query_maxlength |

`contents` expands all corpus fields into each result, including title, text,
sentences and other custom fields. Reserved `id`, `score`, and `embedding` are
controlled by their return selections. `score` is always the PAT reranking score,
replacing any corpus score. `embedding` is the original indexed document vector
as a list of floats, not a contextual PAT embedding. It is disabled by default.

With the default return selection, results look like:

```python
[[{'id': '123', 'score': 0.94, 'title': '...', 'text': '...', ...}, ...]]
```

A string query returns one ranked list; a query list returns one ranked list per
query. Each list position owns an independent cache. Keep positions stable
across turns and call `reset()` between questions. Instances are stateful and
must not serve concurrent questions. `beams` controls cache streams, not query
generation; the caller supplies each beam's query.

Every query retrieves 508 candidates. Initially all 508 are reranked. Later,
the cached documents are placed first, followed by the first `508 - cache_size`
unique new candidates in FAISS order, excluding cached IDs. All 508 are reranked
and the cache is refreshed. The checkpoint must support 508 positions. Scores
are probabilities for classifiers and similarities for embedding checkpoints;
ties retain input order. `cache_size=0` disables retention.

Requires an exact `IndexFlatIP` with at least 508 unique IDs and a matching
corpus. All corpus records require unique `id` values across files (normalized
to strings); remaining fields are preserved. Index IDs must exist in the corpus.
Compressed indexes are rejected because PAT requires original stored vectors.
Use the same retrieval encoder and preprocessing used to build the index and
train PAT. Dimension checks cannot detect different models of the same size.
Load trusted checkpoint and metadata files.

KiRAG defaults to `--document_retrieval pat --pat_config pat_retrieval/pat_retrieval_config.json`.
In PAT mode the configuration controls index, corpus, encoder, PAT_BERT, top-k
and beams; the legacy dense-retrieval path/encoder arguments apply only to
`--document_retrieval dense`. KiRAG supports one or two beams and requires `id`
and `score`. If supplied, `--num_beams` must agree with the config. KiRAG resets
caches per question, uses the configured top-k per turn, and retains its existing
cross-turn document accumulation and final triple-based scoring. Evaluation
scripts consume the same output format. `get_documents()` always loads full
corpus contents, independently of the retrieval return selection.

Validation: `python3 -m unittest discover -s tests -p 'test_pat_retrieval.py'`.
Full inference requires NumPy, FAISS, PyTorch, Transformers, retrieval encoder
weights, index/corpus files and the specified PAT checkpoint.

Set `"debug": true` in the config to print structured data-flow records to stdout
(prefixed `[pat_retrieval]`). Set it to `false` for quiet operation (the code
also defaults to false when omitted). Stages report initialization, cache resets,
query/vector shapes, FAISS search, reconstructed vector shapes, cache overlap,
selected new candidates, PAT scoring, updated cache size, and returned IDs,
scores and field names. Full query text, document contents and vector values
are not dumped. Each query stream is identified by `stream`.

```text
queries -> query_encoded -> faiss_search -> vectors_reconstructed
        -> candidates (cached + deduplicated new)
        -> pat_rerank_start -> pat_rerank_complete -> returned
```
