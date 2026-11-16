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
        self.cache_size = config.cache_size
        self.output_count = config.return_top_k
        self.beams = config.beams
        self.return_items = config.return_items
        self.index = faiss.read_index(config.index_file)
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
        self.reset()

    def reset(self):
        """Discard question-local caches (one independent cache per beam position)."""
        self._caches = {}

    def _rank(self, query, fresh, stream):
        """Keep cached positions first, then unique new candidates in FAISS order."""
        import math
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
        scores = self.reranker.score([query], [candidates])[0]
        if len(scores) != len(candidates) or not all(math.isfinite(s) for s in scores):
            raise ValueError('PAT returned invalid scores')
        ranked = sorted((dict(doc, score=float(score)) for doc, score in zip(candidates, scores)),
                        key=lambda doc: doc['score'], reverse=True)
        self._caches[stream] = ranked[:self.cache_size]
        return [self._format_document(doc) for doc in ranked[:self.output_count]]

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
        if len(queries) > self.beams:
            raise ValueError('Query count exceeds configured beams')
        vectors = np.ascontiguousarray(self.encoder.encode(queries), dtype=np.float32)
        if vectors.shape != (len(queries), self.index.d) or not np.isfinite(vectors).all():
            raise ValueError('Invalid query vectors for FAISS index')
        scores, rows = self.index.search(vectors, self.candidate_count)
        if (rows < 0).any() or not np.isfinite(scores).all():
            raise ValueError('FAISS returned missing documents or invalid scores')
        results = []
        for stream, (query, query_rows, query_scores) in enumerate(zip(vectors, rows, scores)):
            documents = self.index.reconstruct_batch(np.ascontiguousarray(query_rows, dtype=np.int64))
            if not np.isfinite(documents).all():
                raise ValueError('Nonfinite indexed document vectors')
            fresh = [{'id': self.ids[int(row)], 'vector': vector, 'faiss_score': float(score)}
                     for row, vector, score in zip(query_rows, documents, query_scores)]
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
