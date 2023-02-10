import json
from collections import abc
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Type, TypeVar, Union

from typing_extensions import TypeAlias

from .reg_utils import CallableRegistry

IdType: TypeAlias = Union[str, int]

dataset_registry: CallableRegistry["Dataset"] = CallableRegistry()


class Container:
    __slots__: Tuple[str, ...]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def fields(cls) -> Tuple[str, ...]:
        return tuple(getattr(cls, "__annotations__", {}).keys())

    @classmethod
    def _recursive(cls, obj):
        if hasattr(obj, "to_json"):
            return obj.to_json()
        if isinstance(obj, abc.Mapping):
            return {k: cls._recursive(v) for k, v in obj.items()}
        elif isinstance(obj, abc.Iterable) and not isinstance(obj, str):
            return [cls._recursive(v) for v in obj]
        else:
            return obj

    def as_dict(self) -> dict:
        return {k: self._recursive(getattr(self, k)) for k in self.fields()}


class Id(Container):
    __slots__ = ("id",)

    id: Tuple[IdType, ...]

    def __init__(self, id: IdType, *sub_ids):
        self.id = (id, *sub_ids)

    def __str__(self):
        return ".".join(str(i) for i in self.id)

    @classmethod
    def parse(cls, s: Union[IdType, "Id"]) -> "Id":
        return Id(s) if isinstance(s, (str, int)) else s

    @property
    def head(self) -> IdType:
        return self.id[0]

    def __eq__(self, other) -> bool:
        if not isinstance(other, Id):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class Query(Container):
    __slots__ = "qid", "text"

    qid: Id
    text: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.qid = Id.parse(self.qid)


class Document(Container):
    __slots__ = "did", "text", "label", "score"

    did: Id
    text: str
    label: Optional[int]
    score: Optional[float]

    def __init__(self, **kwargs):
        super().__init__(**{"label": None, "score": None, **kwargs})
        self.did = Id.parse(self.did)

    def __hash__(self) -> int:
        return hash((self.did, self.text))

    def new(self, **overrides) -> "Document":
        return type(self)(**{**self.as_dict(), **overrides})

    def judge(self, label: int) -> "Document":
        return self.new(label=label)

    def rank(self, score: float) -> "Document":
        return self.new(score=score)


class Qrel(Container):
    __slots__ = "qid", "did", "label"

    qid: Id
    did: Id
    label: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.qid = Id.parse(self.qid)
        self.did = Id.parse(self.did)


class Dataset(Container):
    __fields__ = "queries", "documents", "qrels"

    queries: List[Query]
    documents: List[Document]
    qrels: List[Qrel]

    def __init__(self, **kwargs):
        super().__init__(
            **{"queries": [], "documents": [], "qrels": [], **kwargs}
        )

    def iter_docs(self) -> Iterable[Document]:
        return (d for d in self.documents)

    def _iter_qrels(self, query: Query, open_set: bool) -> List[Document]:
        qrels = {
            qrel.did.head: qrel.label
            for qrel in self.qrels
            if qrel.qid == query.qid
        }
        docs = [
            doc.judge(qrels.get(doc.did.head, 0))
            for doc in self.documents
            if (open_set or doc.did.head in qrels)
        ]
        return docs

    def iter_close(self, query: Query) -> List[Document]:
        return self._iter_qrels(query=query, open_set=False)

    def iter_open(self, query: Query) -> List[Document]:
        return self._iter_qrels(query=query, open_set=True)


T = TypeVar("T", bound=Container)


def _read_paths(
    root: Path,
    cls: Type[T],
) -> Iterable[T]:

    all_files = [root]

    while len(all_files) > 0:
        path = all_files.pop()
        if path.is_dir():
            all_files.extend(path.iterdir())
            continue

        with open(path) as f:
            for line in f:
                elem = json.loads(line.strip())
                yield cls(**elem)


@dataset_registry.add("from_files")
def from_files(
    docs_path: Union[str, Path],
    queries_path: Optional[Union[str, Path]] = None,
    qrels_path: Optional[Union[str, Path]] = None,
) -> Dataset:

    dataset = Dataset()

    for doc in _read_paths(root=Path(docs_path), cls=Document):
        dataset.documents.append(doc)

    if queries_path is not None:
        for query in _read_paths(root=Path(queries_path), cls=Query):
            dataset.queries.append(query)

    if qrels_path is not None:
        for qrel in _read_paths(root=Path(qrels_path), cls=Qrel):
            dataset.qrels.append(qrel)

    return dataset
