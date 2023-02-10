import re
from enum import Enum
from typing import Any, Generic, List, Sequence, TypeVar, Union

from cachetools import LRUCache

from shadow_scholar.cli import safe_import

from .datasets import Document
from .reg_utils import CallableRegistry

with safe_import():
    # dependencies for models
    import numpy as np
    import spacy
    import spacy.tokens
    import torch
    from sentence_transformers import SentenceTransformer, models
    from sklearn.feature_extraction.text import (
        CountVectorizer,
        TfidfTransformer,
    )
    from sklearn.pipeline import Pipeline
    from unidecode import unidecode


ranker_registry: CallableRegistry["BaseRanker"] = CallableRegistry()

OutputType = TypeVar("OutputType", "torch.Tensor", "np.ndarray")


class BaseRanker(Generic[OutputType]):
    """Abstract base class for scorers.

    A scorer is a class that can score a list of slices; it can do
    so by either using some model, or some other (un)supervised
    method."""

    lru_cache: Union[LRUCache, None]

    def __init__(self, cache_size: int = 10_000):
        self.lru_cache = (
            LRUCache(maxsize=cache_size) if cache_size > 0 else None
        )

    def warmup(self, docs: Sequence[Document]):
        """Pre-compute the encodings for the given documents"""
        pass

    def _encode(self, batch: Sequence[Document]) -> OutputType:
        """Transform tokens in the query and target spans into a numerical
        representation that can be scored by a model"""
        raise NotImplementedError()

    def encode(
        self, docs: Sequence[Document], **encode_kwargs: Any
    ) -> List[OutputType]:
        batch: List[Document] = []
        for doc in docs:
            if self.lru_cache is None or doc not in self.lru_cache:
                batch.append(doc)

        encoded = self._encode(batch, **encode_kwargs) if batch else []
        cache = self.lru_cache or {}
        for i in range(len(batch)):
            cache[batch[i]] = encoded[i]  # pyright: ignore

        return [cache[doc] for doc in docs]

    def _score(
        self, query: str, docs: Sequence[Document]
    ) -> Sequence[Document]:
        """Score a list of slices based on a query"""
        raise NotImplementedError()

    def _sort_key_fn(self, doc: Document) -> float:
        if doc.score is None:
            raise ValueError("Score is None")
        return -doc.score

    def score(self, query: str, docs: Sequence[Document]) -> List[Document]:
        """Score a list of slices based on a query"""
        return sorted(
            [s for s in self._score(query, docs) if s.score],
            key=self._sort_key_fn,
        )


@ranker_registry.add("tfidf")
class TfIdfRanker(BaseRanker["np.ndarray"]):
    def __init__(
        self,
        cache_size: int = 1024,
        lower: bool = True,
        lemma: bool = True,
        stop: bool = False,
        alpha: bool = True,
        ascii: bool = True,
        spacy_model: str = "en_core_web_sm",
    ) -> None:

        super().__init__(cache_size=cache_size)

        self.lower = lower
        self.lemma = lemma
        self.stop = stop
        self.alpha_fn = (
            lambda x: re.match(r"^[a-zA-Z0-9]+$", x) is not None
            if alpha
            else lambda x: True
        )
        self.ascii_fn = unidecode if ascii else lambda x: x
        self.nlp = spacy.load(spacy_model)

    def _word(self, token: "spacy.tokens.token.Token") -> Union[str, None]:
        if self.stop and token.is_stop:
            return None

        text = token.lemma_ if self.lemma else token.text
        text = text.lower() if self.lower else text
        text = self.ascii_fn(text)

        if not self.alpha_fn(text):
            return None

        return text

    def warmup(self, docs: Sequence[Document]):
        content = [
            [
                w
                for w in (self._word(t) for t in self.nlp(doc.text))
                if w is not None
            ]
            for doc in docs
        ]
        vocabulary = sorted(set(w for sl in content for w in sl))
        cv = CountVectorizer(
            vocabulary=vocabulary, analyzer=lambda x: x  # pyright: ignore
        )
        tfidf = TfidfTransformer()
        self.pipeline = Pipeline([("count", cv), ("tfidf", tfidf)])
        self.pipeline.fit(content)

    def _encode(self, docs: Sequence[Document]) -> List["np.ndarray"]:
        content = [
            [
                w
                for w in (self._word(t) for t in self.nlp(doc.text))
                if w is not None
            ]
            for doc in docs
        ]
        return self.pipeline.transform(content).toarray()

    def _score(
        self, query: str, docs: Sequence[Document]
    ) -> Sequence[Document]:
        query_vec, *_ = self.encode([Document(text=query, did=0)])
        corpus_vec = np.vstack(self.encode(docs))

        scores = np.dot(corpus_vec, query_vec).tolist()
        scored_docs = [d.rank(score=s) for d, s in zip(docs, scores)]

        return scored_docs


class PoolingModes(Enum):
    MEAN = {
        "pooling_mode_cls_token": False,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_tokens": True,
        "pooling_mode_mean_sqrt_len_tokens": False,
    }
    MAX = {
        "pooling_mode_cls_token": False,
        "pooling_mode_max_tokens": True,
        "pooling_mode_mean_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
    }
    CLS = {
        "pooling_mode_cls_token": True,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
    }
    SQRT = {
        "pooling_mode_cls_token": False,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": True,
    }


@ranker_registry.add("dense")
class SentenceTransformerRanker(BaseRanker["torch.Tensor"]):
    model: SentenceTransformer

    def __init__(
        self,
        model_name_or_path: str = "facebook/contriever",
        device: str = "cpu",
        batch_size: int = 512,
        norm_query: bool = False,
        norm_doc: bool = False,
        pooling: Union[None, PoolingModes, str] = None,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        if pooling is None:
            model = SentenceTransformer(model_name_or_path=model_name_or_path)
        else:
            if isinstance(pooling, str):
                pooling = PoolingModes[pooling.upper()]

            enc = models.Transformer(model_name_or_path=model_name_or_path)
            pool = models.Pooling(
                enc.get_word_embedding_dimension(), **pooling.value
            )

            model = SentenceTransformer(modules=[enc, pool])

        self.model = model.to(device)
        self.norm_query = norm_query
        self.norm_doc = norm_doc
        self.batch_size = batch_size

    def _encode(  # type: ignore[override]
        self, batch: Sequence[Document], norm: bool
    ) -> "torch.Tensor":
        encoding = self.model.encode(
            sentences=[doc.text for doc in batch],
            convert_to_tensor=True,
            normalize_embeddings=norm,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )
        assert isinstance(encoding, torch.Tensor)
        return encoding.to("cpu")

    def _score(
        self, query: str, docs: Sequence[Document]
    ) -> Sequence[Document]:

        query_vec, *_ = self.encode(
            [Document(text=query, did=0)], norm=self.norm_doc
        )
        corpus_vec = torch.vstack(self.encode(docs, norm=self.norm_query))

        scores = corpus_vec @ query_vec.reshape(-1, 1)

        scores = scores.squeeze().tolist()
        scored_docs = [d.rank(score=s) for d, s in zip(docs, scores)]

        return scored_docs
