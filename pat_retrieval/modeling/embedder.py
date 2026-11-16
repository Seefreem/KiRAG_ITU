"""Contextual BERT query/passage embeddings scored by unnormalized dot product."""

import torch

from .bert import BertDocumentBackbone


class BertDocumentEmbedder(BertDocumentBackbone):
    model_type = 'embedder'

    def forward(self, query_embeddings, document_embeddings):
        query, passages = self.contextual_embeddings(query_embeddings, document_embeddings)
        return torch.einsum('bh,bnh->bn', query, passages)
