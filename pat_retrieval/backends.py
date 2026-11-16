"""FAISS, retrieval encoder, and trained BERT adapters for the evaluation loop."""

import numpy as np
import torch




class QueryEncoder:
    """Encode queries locally with the retrieval model recorded in the index."""
    def __init__(self, manifest, device, query_maxlength=512):
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.retriever = manifest['retriever']
        if self.retriever not in ('E5Retriever', 'BGERetriever'):
            raise ValueError(f'Unsupported retrieval encoder: {self.retriever}')
        self.query_maxlength = query_maxlength
        self.tokenizer = AutoTokenizer.from_pretrained(manifest['tokenizer'])
        if self.tokenizer.pad_token_id is None:
            raise ValueError('Retrieval tokenizer requires an existing padding token')
        self.model = AutoModel.from_pretrained(manifest['model']).to(device).eval()

    def encode(self, questions):
        """Encode question strings as CPU float32 vectors [batch, query_dimension]."""
        prefix = ('query: ' if self.retriever == 'E5Retriever'
                  else 'Represent this sentence for searching relevant passages: ')
        inputs = self.tokenizer([prefix + question for question in questions],
                                padding=True, truncation=True,
                                max_length=self.query_maxlength, return_tensors='pt')
        # The original collator forwards only these two fields (in particular,
        # it discards tokenizer-supplied token_type_ids).
        inputs = {key: inputs[key].to(self.device)
                  for key in ('input_ids', 'attention_mask')}
        with torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            if self.retriever == 'E5Retriever':
                mask = inputs['attention_mask'].bool()
                vectors = hidden.masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=1)
                vectors = vectors / mask.sum(dim=1, keepdim=True)
            else:
                vectors = hidden[:, 0]
            # Both original E5Encoder and BGEEncoder normalize internally.
            # norm_query=False in their wrapper only skips a SECOND normalize.
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            return vectors.detach().to(device='cpu', dtype=torch.float32).numpy()


class BertReranker:
    """Score each of 508 passages using a classifier or contextual embedder."""
    def __init__(self, checkpoint, device):
        from .modeling.bert import BertDocumentClassifier

        self.device = device
        from .modeling.embedder import BertDocumentEmbedder

        from .modeling.aligned_embedder import BertAlignedEmbedder

        model_class = {'embedder': BertDocumentEmbedder,
                       'aligned_embedder': BertAlignedEmbedder}.get(
                           checkpoint.get('model_type'), BertDocumentClassifier)
        self.model = model_class.from_checkpoint(checkpoint).to(device).eval()
        self.score_kind = 'similarity' if self.model.model_type in ('embedder', 'aligned_embedder') else 'probability'
        if self.model.document_count != 508:
            raise ValueError('This evaluation requires a BERT checkpoint with exactly 508 document positions')

    @classmethod
    def from_model(cls, model, device):
        """Wrap the current training weights without reconstructing a checkpoint."""
        if model.document_count != 508:
            raise ValueError('Retrieval evaluation requires 508 document positions')
        instance = cls.__new__(cls)
        instance.device, instance.model = device, model
        instance.score_kind = 'similarity' if model.model_type in ('embedder', 'aligned_embedder') else 'probability'
        return instance

    def score(self, queries, candidates):
        """Return [batch][candidate] scores, preserving input order.

        Classifiers return sigmoid probabilities; embedders return raw dot
        products. score_kind tells result formatting which range is valid.
        """
        vectors = np.stack([[candidate['vector'] for candidate in documents] for documents in candidates])
        query_tensor = torch.from_numpy(np.asarray(queries, dtype=np.float32)).to(self.device)
        document_tensor = torch.from_numpy(vectors.astype(np.float32, copy=False)).to(self.device)
        with torch.inference_mode():
            logits = self.model(query_tensor, document_tensor)
            scores = logits if self.score_kind == 'similarity' else logits.sigmoid()
            return scores.cpu().tolist()
