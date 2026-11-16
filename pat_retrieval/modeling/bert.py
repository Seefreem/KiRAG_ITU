"""Contextual document backbone and classifier using Hugging Face BERT inputs_embeds.

Only raw word embeddings of special tokens are looked up manually. BERT adds
position/type embeddings and applies its embedding LayerNorm exactly once.
"""

import torch
from torch import nn
from transformers import BertConfig, BertModel

from .adaptors import ADAPTORS


class BertDocumentBackbone(nn.Module):
    def __init__(self, bert, query_dim, document_dim, cls_token_id, sep_token_id,
                 adaptor='linear', shared_adaptor=True):
        super().__init__()
        self.bert = bert
        if bert.config.is_decoder or bert.config.type_vocab_size < 2:
            raise ValueError('Expected encoder BERT with at least two token types')
        for token_id in (cls_token_id, sep_token_id):
            if token_id is None or not 0 <= token_id < bert.config.vocab_size:
                raise ValueError('Tokenizer requires valid CLS and SEP tokens in the BERT vocabulary')
        
        self.document_count = bert.config.max_position_embeddings - 4
        if self.document_count < 1:
            raise ValueError('BERT position capacity must exceed four')
        self.adaptor = ADAPTORS[adaptor](query_dim, document_dim, bert.config.hidden_size,
                                       shared=shared_adaptor)
        self.cls_token_id, self.sep_token_id = cls_token_id, sep_token_id
        self.training_mode = 'both'
        self.spec = dict(query_dim=query_dim, document_dim=document_dim,
                         cls_token_id=cls_token_id, sep_token_id=sep_token_id,
                         adaptor=adaptor, shared_adaptor=shared_adaptor)

    def set_training_mode(self, mode):
        if mode not in ('adaptor', 'bert', 'both'):
            raise ValueError(f'Unknown training mode {mode}')
        self.training_mode = mode
        self.bert.requires_grad_(mode in ('bert', 'both'))
        self.adaptor.requires_grad_(mode in ('adaptor', 'both'))
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.training_mode == 'adaptor':
            self.bert.eval()  # Disable frozen-backbone dropout, retain input gradients.
        return self

    def contextual_embeddings(self, query_embeddings, document_embeddings):
        if query_embeddings.ndim != 2 or document_embeddings.ndim != 3:
            raise ValueError('Expected query [B,Dq] and documents [B,N,Dd]')
        batch_size, count, _ = document_embeddings.shape
        if count != self.document_count or query_embeddings.shape[0] != batch_size:
            raise ValueError(f'Each query must have exactly {self.document_count} documents')
        query, documents = self.adaptor(query_embeddings, document_embeddings)
        
        special_ids = torch.tensor([self.cls_token_id, self.sep_token_id], device=query.device)
        embedding_layer = self.bert.get_input_embeddings()
        specials = embedding_layer(special_ids)
        cls = specials[0].view(1, 1, -1).expand(batch_size, 1, -1)
        sep = specials[1].view(1, 1, -1).expand(batch_size, 1, -1)
        # [CLS], query, [SEP], doc_1, ..., doc_N, [SEP]
        inputs = torch.cat([cls, query.unsqueeze(1), sep, documents, sep], dim=1)
        token_types = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
        token_types[:, :3] = 0
        outputs = self.bert(inputs_embeds=inputs, token_type_ids=token_types,
                            attention_mask=torch.ones_like(token_types), return_dict=True)
        return outputs.last_hidden_state[:, 1, :], outputs.last_hidden_state[:, 3:-1, :]

    def checkpoint(self):
        return {'format_version': 2, 'model_type': self.model_type, 'bert_config': self.bert.config.to_dict(),
                'model_spec': self.spec, 'state_dict': self.state_dict(),
                'training_mode': self.training_mode}

    @classmethod
    def from_checkpoint(cls, checkpoint):
        # Offline reconstruction: all pretrained and finetuned weights are saved.
        if checkpoint.get('model_type', 'classifier') != cls.model_type:
            raise ValueError('Checkpoint architecture does not match the requested model')
        if checkpoint['format_version'] != 2:
            raise ValueError('Unsupported checkpoint format')
        bert = BertModel(BertConfig.from_dict(checkpoint['bert_config']), add_pooling_layer=False)
        # Reconstruct the custom model
        model = cls(bert, **checkpoint['model_spec'])
        # Load the saved weights and training mode.
        model.load_state_dict(checkpoint['state_dict'])
        model.set_training_mode(checkpoint['training_mode'])
        return model


class BertDocumentClassifier(BertDocumentBackbone):
    model_type = 'classifier'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def set_training_mode(self, mode):
        super().set_training_mode(mode)
        self.classifier.requires_grad_(True)

    def forward(self, query_embeddings, document_embeddings):
        _, documents = self.contextual_embeddings(query_embeddings, document_embeddings)
        return self.classifier(documents).squeeze(-1)
