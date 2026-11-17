"""Stateful retrieval for one question at a time; call reset between questions."""
from copy import deepcopy
import pickle

from .config import RetrievalConfig, DEFAULT_CHECKPOINT
from .corpus import FileCorpus


class PATRetrievalPipeline:
    candidate_count = 508

    def __init__(self, config):
        """Accept a RetrievalConfig instance or the path to a JSON configuration."""
        import faiss
        import torch
        from .backends import BertReranker, QueryEncoder

        self.config = config if isinstance(config, RetrievalConfig) else RetrievalConfig.from_file(config)
        config = self.config
        self.debug = config.debug
        self.use_cached_query_for_pat_bert = config.use_cached_query_for_pat_bert
        self.cache_size = config.cache_size
        self.output_count = config.return_top_k
        self.beams = config.beams
        self.return_items = config.return_items
        self.index = faiss.read_index(config.index_file)
        # The query encoder L2-normalizes its embeddings. If the stored document embeddings are
        # also L2-normalized—as expected for the E5 index—inner product equals cosine similarity.
        if not isinstance(self.index, faiss.IndexFlatIP):
            raise ValueError('PAT requires IndexFlatIP with exact stored document vectors')
        with open(config.index_meta_file, 'rb') as source:
            self.ids = [str(pid) for pid in pickle.load(source)]
        if (len(self.ids) != self.index.ntotal or len(set(self.ids)) != len(self.ids)
                or self.index.ntotal < self.candidate_count):
            raise ValueError('Index needs matching unique metadata and at least 508 documents')
        self.corpus = FileCorpus(config.corpus_files)
        if not set(self.ids).issubset(self.corpus.documents):
            raise ValueError('FAISS metadata IDs are missing from the configured corpus')
        bert = config.pat_bert
        self.reranker = BertReranker(torch.load(bert.checkpoint, map_location='cpu', weights_only=False), bert.device)
        spec = self.reranker.model.spec
        if spec['query_dim'] != self.index.d or spec['document_dim'] != self.index.d:
            raise ValueError('PAT checkpoint dimensions do not match the FAISS index')
        encoder = config.encoder
        self.encoder = QueryEncoder(dict(retriever=encoder.retriever, model=encoder.model, tokenizer=encoder.tokenizer),
                                    encoder.device, encoder.query_maxlength)
        self._trace('loaded', index_file=config.index_file, metric='inner_product',
                    indexed_documents=self.index.ntotal, embedding_dimension=self.index.d,
                    corpus_files=list(config.corpus_files), checkpoint=bert.checkpoint,
                    cache_size=self.cache_size, beams=self.beams, use_cached_query_for_pat_bert=self.use_cached_query_for_pat_bert,
                    return_top_k=self.output_count, return_items=list(self.return_items))
        self.reset()

    def _trace(self, stage, **data):
        """Print one structured record per stage, without dumping vectors/text."""
        if getattr(self, 'debug', False):
            import json
            print('[pat_retrieval] ' + json.dumps(dict(stage=stage, **data)), flush=True)

    def reset(self):
        """Discard question-local caches (one independent cache per beam position).
        # Structure: {stream: [cached_document, ...]}.
        # stream is the integer query/beam position (0, 1, ...).
        # Each list contains at most cache_size documents, sorted by descending
        # PAT score from that stream's most recent reranking. IDs are unique
        # within a stream; different streams may cache the same document.
        # A cached_document contains:
        #   id: str                  -- corpus document ID
        #   vector: ndarray[D]       -- original FAISS vector (float32)
        #   faiss_score: float       -- similarity when first admitted to cache
        #   score: float             -- latest PAT reranking score
        # Titles/text remain in the corpus, not in this internal cache.
        """
        self._caches = {}
        self._original_queries = {}  # {beam position: first query since reset()}
        self._trace('cache_reset')

    def _select_queries(self, queries):
        """Select PAT_BERT queries; FAISS always uses the incoming queries."""
        if len(queries) > self.beams:
            raise ValueError('Query count exceeds configured beams')
        if any(not isinstance(query, str) or not query.strip() for query in queries):
            raise ValueError('Queries must be nonempty strings')
        for stream, query in enumerate(queries):
            self._original_queries.setdefault(stream, query)
        selected = ([self._original_queries[stream] for stream in range(len(queries))]
                    if self.use_cached_query_for_pat_bert else list(queries))
        self._trace('query_selection', use_cached_query_for_pat_bert=self.use_cached_query_for_pat_bert,
                    streams=list(range(len(queries))),
                    differs_from_provided=[query != provided for query, provided in zip(selected, queries)])
        return selected

    def _encode_queries(self, incoming):
        """Return FAISS vectors from incoming queries and PAT vectors by policy."""
        pat_queries = self._select_queries(incoming)
        faiss_vectors = self.encoder.encode(incoming)
        pat_vectors = (faiss_vectors if pat_queries == incoming
                       else self.encoder.encode(pat_queries))
        return faiss_vectors, pat_vectors

    def _rank(self, query, fresh, stream):
        """Keep cached positions first, then unique new candidates in FAISS order."""
        import math
        # Missing stream means its first iteration: no retained documents yet.
        cached = self._caches.get(stream, [])
        candidates = list(cached)
        seen = {doc['id'] for doc in cached}
        for document in fresh:
            if document['id'] not in seen:
                candidates.append(document)
                seen.add(document['id'])
            if len(candidates) == self.candidate_count:
                break
        # A unique 508-result search always fills the available positions, even
        # when every cached document occurs again in the fresh search.
        if len(candidates) != self.candidate_count:
            raise ValueError('PAT needs exactly 508 unique document candidates')
        cached_ids = {doc['id'] for doc in cached}
        self._trace('candidates', stream=stream, fresh_count=len(fresh), cached_count=len(cached),
                    fresh_unique_count=len({doc['id'] for doc in fresh}),
                    fresh_cache_overlap=sum(doc['id'] in cached_ids for doc in fresh),
                    selected_new_count=len(candidates) - len(cached), pat_input_count=len(candidates))
        self._trace('pat_rerank_start', stream=stream, query_shape=[1, len(query)],
                    document_shape=[1, len(candidates), len(candidates[0]['vector'])])
        scores = self.reranker.score([query], [candidates])[0]
        if len(scores) != len(candidates) or not all(math.isfinite(s) for s in scores):
            raise ValueError('PAT returned invalid scores')
        ranked = sorted((dict(doc, score=float(score)) for doc, score in zip(candidates, scores)),
                        key=lambda doc: doc['score'], reverse=True)
        # Replace only this beam's cache; score is refreshed for every retained
        # document, while vector and historical faiss_score are preserved.
        self._caches[stream] = ranked[:self.cache_size]
        self._trace('pat_rerank_complete', stream=stream, score_min=min(scores), score_max=max(scores),
                    cache_count=len(self._caches[stream]))
        result = [self._format_document(doc) for doc in ranked[:self.output_count]]
        self._trace('returned', stream=stream, count=len(result),
                    top_documents=[{'id': doc['id'], 'score': doc['score']} for doc in ranked[:self.output_count]],
                    return_items=list(self.return_items), fields=sorted(result[0]) if result else [])
        return result

    def _format_document(self, document):
        result = {}
        if 'contents' in self.return_items:
            result.update(self.get_documents([document['id']])[0])
            # Reserved fields always reflect the output selection and reranking.
            for key in ('id', 'score', 'embedding'):
                result.pop(key, None)
        if 'id' in self.return_items:
            result['id'] = document['id']
        if 'score' in self.return_items:
            result['score'] = document['score']
        if 'embedding' in self.return_items:
            vector = document['vector']
            result['embedding'] = vector.tolist() if hasattr(vector, 'tolist') else list(vector)
        return result

    def __call__(self, queries, topk=None, **kwargs):
        """Return configured fields; list positions identify stable reasoning beams."""
        import numpy as np
        if topk is not None and topk != self.output_count:
            raise ValueError('topk must match return_top_k in the pipeline configuration')
        single = isinstance(queries, str)
        queries = [queries] if single else list(queries)
        if not queries:
            return []
        self._trace('queries', count=len(queries), character_counts=[len(query) for query in queries])
        faiss_vectors, pat_vectors = self._encode_queries(queries)
        vectors = np.ascontiguousarray(faiss_vectors, dtype=np.float32)
        pat_vectors = np.ascontiguousarray(pat_vectors, dtype=np.float32)
        if vectors.shape != (len(queries), self.index.d) or not np.isfinite(vectors).all():
            raise ValueError('Invalid query vectors for FAISS index')
        if pat_vectors.shape != vectors.shape or not np.isfinite(pat_vectors).all():
            raise ValueError('Invalid PAT_BERT query vectors')
        self._trace('query_encoded', faiss_shape=list(vectors.shape),
                    pat_shape=list(pat_vectors.shape), dtype=str(vectors.dtype))
        # distances, ids = index.search(query_vectors, k=3)
        # Both arrays have shape (number_of_queries, k). For query i, ids[i, j] identifies the j-th ranked result,
        # and distances[i, j] gives its distance or similarity score.
        # Descending order of scores is expected for inner product similarity, which is the default for E5.
        scores, rows = self.index.search(vectors, self.candidate_count)
        if (rows < 0).any() or not np.isfinite(scores).all():
            raise ValueError('FAISS returned missing documents or invalid scores')
        self._trace('faiss_search', metric='inner_product', result_shape=list(rows.shape))
        results = []
        for stream, (query, retrieved_ids, query_scores) in enumerate(zip(pat_vectors, rows, scores)):
            documents = self.index.reconstruct_batch(np.ascontiguousarray(retrieved_ids, dtype=np.int64))
            if not np.isfinite(documents).all():
                raise ValueError('Nonfinite indexed document vectors')
            self._trace('vectors_reconstructed', stream=stream, shape=list(documents.shape),
                        faiss_score_max=float(query_scores.max()), faiss_score_min=float(query_scores.min()))
            fresh = [{'id': self.ids[int(row)], 'vector': vector, 'faiss_score': float(score)}
                     for row, vector, score in zip(retrieved_ids, documents, query_scores)]
            results.append(self._rank(query, fresh, stream))
        return results[0] if single else results

    def get_documents(self, docids):
        """Resolve IDs using a corpus with get_document(id), or an ID-keyed mapping."""
        items = (sorted(docids.items(), key=lambda item: item[1], reverse=True)
                 if isinstance(docids, dict) else [(pid, None) for pid in docids])
        documents = []
        for pid, score in items:
            source = (self.corpus.get_document(pid) if hasattr(self.corpus, 'get_document')
                      else self.corpus[pid])
            document = deepcopy(source)
            if score is not None:
                document['score'] = float(score)
            documents.append(document)
        return documents
