"""Document storage independent of KiRAG; JSON arrays or JSONL shards."""
import json
from pathlib import Path


class FileCorpus:
    def __init__(self, files):
        self.documents = {}
        for filename in files:
            path = Path(filename)
            with path.open(encoding='utf-8') as source:
                rows = (json.loads(line) for line in source if line.strip()) if path.suffix == '.jsonl' else json.load(source)
                if not hasattr(rows, '__iter__') or isinstance(rows, dict):
                    raise ValueError(f'{path}: expected a JSON document array or JSONL')
                for row in rows:
                    pid = str(row['id'])
                    if not pid or pid in self.documents:
                        raise ValueError(f'{path}: empty or duplicate document ID {pid!r}')
                    self.documents[pid] = dict(row, id=pid)

    def get_document(self, pid):
        return self.documents[str(pid)]
