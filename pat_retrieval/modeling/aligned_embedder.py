"""Contextual embeddings projected back into the original retrieval space."""

import torch
from torch import nn
from torch.nn import functional as F

from .bert import BertDocumentBackbone


class BertAlignedEmbedder(BertDocumentBackbone):
    model_type = 'aligned_embedder'

    def __init__(self, bert, query_dim, document_dim, cls_token_id, sep_token_id,
                 adaptor='linear', shared_adaptor=True):
        if query_dim != document_dim:
            raise ValueError('Aligned embeddings require equal query and passage dimensions')
        super().__init__(bert, query_dim, document_dim, cls_token_id, sep_token_id,
                         adaptor=adaptor, shared_adaptor=shared_adaptor)
        # One shared output mapping keeps query and passage outputs comparable.
        self.output_adaptor = nn.Linear(bert.config.hidden_size, query_dim)

    def set_training_mode(self, mode):
        super().set_training_mode(mode)
        # The new output mapping is learned in every training mode.
        self.output_adaptor.requires_grad_(True)

    def retrieval_embeddings(self, query_embeddings, document_embeddings):
        query, passages = self.contextual_embeddings(query_embeddings, document_embeddings)
        return (F.normalize(self.output_adaptor(query), p=2, dim=-1),
                F.normalize(self.output_adaptor(passages), p=2, dim=-1))

    def forward(self, query_embeddings, document_embeddings, return_embeddings=False):
        query, passages = self.retrieval_embeddings(query_embeddings, document_embeddings)
        scores = torch.einsum('bd,bnd->bn', query, passages)
        if return_embeddings:
            return {'scores': scores, 'query': query, 'passages': passages}
        return scores
