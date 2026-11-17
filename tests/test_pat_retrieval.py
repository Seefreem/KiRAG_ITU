import unittest

from pat_retrieval import PATRetrievalPipeline


class RecordingReranker:
    def __init__(self):
        self.inputs = []

    def score(self, queries, candidates):
        self.inputs.append(candidates[0])
        return [[float(doc['id']) for doc in candidates[0]]]


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = PATRetrievalPipeline.__new__(PATRetrievalPipeline)
        self.pipeline.reranker = RecordingReranker()
        self.pipeline.cache_size = 58
        self.pipeline.output_count = 10
        self.pipeline.return_items = ('id', 'score')
        self.pipeline.reset()

    def fresh(self, start=0):
        return [{'id': str(i), 'vector': [i]} for i in range(start, start + 508)]

    def test_rollover_deduplicates_and_returns_ten(self):
        result = self.pipeline._rank([0], self.fresh(), 0)
        self.assertEqual([doc['id'] for doc in result], [str(i) for i in range(507, 497, -1)])
        cached_ids = [doc['id'] for doc in self.pipeline._caches[0]]
        self.assertEqual(len(cached_ids), 58)
        self.pipeline._rank([1], self.fresh(400), 0)
        inputs = self.pipeline.reranker.inputs[-1]
        ids = [doc['id'] for doc in inputs]
        self.assertEqual(ids[:58], cached_ids)
        self.assertEqual(len(ids), 508)
        self.assertEqual(len(set(ids)), 508)
        expected_new = [str(i) for i in range(400, 908) if str(i) not in cached_ids][:450]
        self.assertEqual(ids[58:], expected_new)
        self.assertEqual(self.pipeline._caches[0][0]['id'], expected_new[-1])

    def test_beams_are_independent_and_reset_clears_all(self):
        self.pipeline._rank([0], self.fresh(), 0)
        self.pipeline._rank([0], self.fresh(1000), 1)
        self.assertTrue(all(int(doc['id']) >= 1000 for doc in self.pipeline._caches[1]))
        self.assertTrue(all(int(doc['id']) < 508 for doc in self.pipeline._caches[0]))
        self.pipeline.reset()
        self.assertEqual(self.pipeline._caches, {})

    def test_rejects_incomplete_unique_input(self):
        with self.assertRaisesRegex(ValueError, '508 unique'):
            self.pipeline._rank([0], self.fresh()[:500], 0)

    def test_document_resolution_preserves_corpus(self):
        self.pipeline.corpus = {'a': {'id': 'a', 'text': 'hello'}}
        self.assertEqual(self.pipeline.get_documents({'a': 0.8})[0]['score'], 0.8)
        self.assertNotIn('score', self.pipeline.corpus['a'])


class ConfigurationTests(unittest.TestCase):
    def test_config_paths_and_corpus_shards(self):
        import json
        from pathlib import Path
        import tempfile
        from pat_retrieval import RetrievalConfig
        from pat_retrieval.corpus import FileCorpus
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a.json').write_text(json.dumps([{'id': 1, 'title': 'A', 'text': 'hello', 'extra': 42}]))
            (root / 'b.jsonl').write_text(json.dumps({'id': '2', 'text': 'world'}) + '\n')
            data = dict(index_file='custom.faiss', index_meta_file='custom.meta',
                        corpus_files=['a.json', 'b.jsonl'], return_top_k=7, cache_size=0, beams=2,
                        return_items=['id', 'score', 'contents', 'embedding'])
            (root / 'config.json').write_text(json.dumps(data))
            config = RetrievalConfig.from_file(root / 'config.json')
            self.assertEqual(config.index_file, str((root / 'custom.faiss').resolve()))
            self.assertEqual(config.return_top_k, 7)
            corpus = FileCorpus(config.corpus_files)
            pipeline = PATRetrievalPipeline.__new__(PATRetrievalPipeline)
            pipeline.corpus, pipeline.return_items = corpus, config.return_items
            result = pipeline._format_document(dict(id='1', score=0.75, vector=[1., 2.]))
            self.assertEqual(result, dict(id='1', score=0.75, title='A', text='hello', extra=42, embedding=[1., 2.]))
            pipeline.return_items = ('id',)
            self.assertEqual(pipeline._format_document(dict(id='1')), {'id': '1'})
            pipeline.return_items = ('contents',)
            self.assertNotIn('id', pipeline._format_document(dict(id='1')))
            with self.assertRaises(ValueError):
                FileCorpus([root / 'a.json', root / 'a.json'])

    def test_invalid_configuration(self):
        from pat_retrieval import RetrievalConfig
        base = dict(index_file='i', index_meta_file='m', corpus_files=('c',))
        for change in (dict(cache_size=508), dict(cache_size=-1), dict(return_top_k=0),
                       dict(beams=0), dict(return_items=('unknown',)), dict(return_items='id'),
                       dict(corpus_files='corpus.json'), dict(debug='true'), dict(use_cached_query_for_pat_bert='true')):
            with self.subTest(change=change), self.assertRaises(ValueError):
                RetrievalConfig(**(base | change))

    def test_configurable_cache_and_output_sizes(self):
        pipeline = PATRetrievalPipeline.__new__(PATRetrievalPipeline)
        pipeline.reranker = RecordingReranker()
        pipeline.cache_size, pipeline.output_count = 3, 7
        pipeline.return_items = ('id', 'score')
        pipeline.reset()
        fresh = [{'id': str(i), 'vector': [i]} for i in range(508)]
        self.assertEqual(len(pipeline._rank([0], fresh, 0)), 7)
        self.assertEqual(len(pipeline._caches[0]), 3)
        pipeline.cache_size = 0
        pipeline._rank([0], fresh, 0)
        self.assertEqual(pipeline._caches[0], [])


class DebugTests(unittest.TestCase):
    def test_trace_is_opt_in_and_describes_cache_rollover(self):
        from contextlib import redirect_stdout
        from io import StringIO
        import json
        pipeline = PATRetrievalPipeline.__new__(PATRetrievalPipeline)
        pipeline.reranker = RecordingReranker()
        pipeline.cache_size, pipeline.output_count = 58, 10
        pipeline.return_items = ('id', 'score')
        pipeline.debug = False
        output = StringIO()
        fresh = [{'id': str(i), 'vector': [i]} for i in range(508)]
        with redirect_stdout(output):
            pipeline.reset()
            pipeline._rank([0], fresh, 0)
        self.assertEqual(output.getvalue(), '')
        pipeline.debug = True
        with redirect_stdout(output):
            pipeline._rank([0], fresh, 0)
        records = [json.loads(line.removeprefix('[pat_retrieval] ')) for line in output.getvalue().splitlines()]
        self.assertEqual([row['stage'] for row in records],
                         ['candidates', 'pat_rerank_start', 'pat_rerank_complete', 'returned'])
        self.assertEqual(records[0]['cached_count'], 58)
        self.assertEqual(records[0]['selected_new_count'], 450)
        self.assertEqual(records[0]['fresh_cache_overlap'], 58)
        self.assertEqual(records[1]['document_shape'], [1, 508, 1])
        self.assertEqual(records[-1]['count'], 10)
        self.assertNotIn('vector', output.getvalue())


class QuerySelectionTests(unittest.TestCase):
    def make_pipeline(self, use_cached_query_for_pat_bert):
        pipeline = PATRetrievalPipeline.__new__(PATRetrievalPipeline)
        pipeline.beams = 2
        pipeline.use_cached_query_for_pat_bert = use_cached_query_for_pat_bert
        pipeline.reset()
        return pipeline

    def test_original_queries_persist_per_beam_and_reset(self):
        pipeline = self.make_pipeline(True)
        self.assertEqual(pipeline._select_queries(['original A', 'original B']), ['original A', 'original B'])
        self.assertEqual(pipeline._select_queries(['updated A', 'updated B']), ['original A', 'original B'])
        self.assertEqual(pipeline._select_queries(['updated again']), ['original A'])
        pipeline.reset()
        self.assertEqual(pipeline._original_queries, {})
        self.assertEqual(pipeline._select_queries(['new question']), ['new question'])

    def test_updated_query_mode_still_remembers_original(self):
        pipeline = self.make_pipeline(False)
        pipeline._select_queries(['original'])
        self.assertEqual(pipeline._select_queries(['updated']), ['updated'])
        self.assertEqual(pipeline._original_queries, {0: 'original'})

    def test_faiss_always_encodes_incoming_query(self):
        class RecordingEncoder:
            def __init__(self):
                self.calls = []

            def encode(self, queries):
                self.calls.append(list(queries))
                return list(queries)  # Distinct stand-ins for encoded vectors.

        for cached in (True, False):
            with self.subTest(cached=cached):
                pipeline = self.make_pipeline(cached)
                pipeline.encoder = RecordingEncoder()
                faiss, pat = pipeline._encode_queries(['original'])
                self.assertIs(faiss, pat)
                faiss, pat = pipeline._encode_queries(['updated'])
                self.assertEqual(faiss, ['updated'])
                self.assertEqual(pat, ['original'] if cached else ['updated'])
                self.assertEqual(pipeline.encoder.calls,
                                 [['original'], ['updated'], ['original']] if cached
                                 else [['original'], ['updated']])

    def test_invalid_queries_do_not_change_originals(self):
        pipeline = self.make_pipeline(True)
        for queries in (['one', 'two', 'three'], ['valid', ''], [None]):
            with self.assertRaises(ValueError):
                pipeline._select_queries(queries)
        self.assertEqual(pipeline._original_queries, {})


if __name__ == '__main__':
    unittest.main()
