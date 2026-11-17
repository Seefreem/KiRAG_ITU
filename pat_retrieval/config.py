"""Validated JSON configuration. Relative data paths resolve beside the config."""
from dataclasses import dataclass, field
import json
from pathlib import Path

DEFAULT_CHECKPOINT = 'checkpoint/bert_pat_checkp_tboth_bs128_ep2_lr2e_6_wbce_pw10_3_skip79000steps/best.pt'


@dataclass(frozen=True)
class PATBertConfig:
    checkpoint: str = DEFAULT_CHECKPOINT
    device: str = 'cuda'


@dataclass(frozen=True)
class EncoderConfig:
    model: str = 'intfloat/e5-large-v2'
    tokenizer: str = 'intfloat/e5-large-v2'
    retriever: str = 'E5Retriever'
    device: str = 'cuda'
    query_maxlength: int = 512

    def __post_init__(self):
        if self.retriever not in ('E5Retriever', 'BGERetriever'):
            raise ValueError('Unsupported retrieval encoder')
        if type(self.query_maxlength) is not int or self.query_maxlength < 1:
            raise ValueError('query_maxlength must be positive')


@dataclass(frozen=True)
class RetrievalConfig:
    index_file: str
    index_meta_file: str
    corpus_files: tuple[str, ...]
    return_top_k: int = 10
    cache_size: int = 58
    beams: int = 1
    debug: bool = False
    return_items: tuple[str, ...] = ('id', 'score', 'contents')
    pat_bert: PATBertConfig = field(default_factory=PATBertConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    def __post_init__(self):
        if type(self.debug) is not bool:
            raise ValueError('debug must be a boolean')
        for name, lower, upper in (('return_top_k', 1, 508), ('cache_size', 0, 507), ('beams', 1, None)):
            value = getattr(self, name)
            if type(value) is not int or value < lower or (upper is not None and value > upper):
                raise ValueError(f'Invalid {name}: {value}')
        if not self.corpus_files or isinstance(self.corpus_files, str):
            raise ValueError('corpus_files must be a nonempty list of paths')
        allowed = {'id', 'score', 'contents', 'embedding'}
        if (not self.return_items or isinstance(self.return_items, str)
                or set(self.return_items) - allowed or len(set(self.return_items)) != len(self.return_items)):
            raise ValueError(f'return_items must select unique names from {sorted(allowed)}')

    @classmethod
    def from_file(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding='utf-8'))
        def resolve(value):
            return str((path.parent / value).resolve())
        for key in ('index_file', 'index_meta_file'):
            data[key] = resolve(data[key])
        if not isinstance(data['corpus_files'], list):
            raise ValueError('corpus_files must be a list')
        data['corpus_files'] = tuple(resolve(value) for value in data['corpus_files'])
        bert = data.get('pat_bert', {})
        bert['checkpoint'] = resolve(bert.get('checkpoint', DEFAULT_CHECKPOINT))
        data['pat_bert'] = PATBertConfig(**bert)
        data['encoder'] = EncoderConfig(**data.get('encoder', {}))
        if 'return_items' in data:
            if not isinstance(data['return_items'], list):
                raise ValueError('return_items must be a list')
            data['return_items'] = tuple(data['return_items'])
        return cls(**data)
