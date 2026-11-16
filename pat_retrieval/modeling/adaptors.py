"""Adaptors are independent of the contextual backbone and training objective."""

from torch import nn


class LinearAdaptor(nn.Module):
    def __init__(self, query_dim, document_dim, hidden_size, shared=True):
        super().__init__()
        if shared and query_dim != document_dim:
            raise ValueError('Shared adaptor requires equal query/document dimensions; use --separate-adaptors')
        self.query = nn.Linear(query_dim, hidden_size)
        self.document = self.query if shared else nn.Linear(document_dim, hidden_size)

    def forward(self, queries, documents):
        return self.query(queries), self.document(documents)


# Add alternative adaptor implementations here without changing the dataset.
ADAPTORS = {'linear': LinearAdaptor}
