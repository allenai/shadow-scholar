from abc import ABC, abstractmethod
from typing import List, Union

from shadow_scholar.cli import safe_import

from .datasets import Document
from .reg_utils import CallableRegistry

with safe_import():
    import spacy


slicer_registry: CallableRegistry["BaseSlicer"] = CallableRegistry()


class BaseSlicer(ABC):
    """Abstract base class for slicers.
    A slicer is a class that can slice a document into a list of slices;
    each slice is a list of spans and/or boxes, alongside the text of
    the slice."""

    @abstractmethod
    def slice_one(self, doc: Document) -> List[Document]:
        ...

    def slice(
        self, seq_or_doc: Union[Document, List[Document]]
    ) -> List[Document]:
        if isinstance(seq_or_doc, Document):
            return self.slice_one(seq_or_doc)
        else:
            return [doc for doc in seq_or_doc for doc in self.slice_one(doc)]


@slicer_registry.add("sent")
class SentenceSlicer(BaseSlicer):
    """A slicer that slices a document into a single sentence."""

    stride: int
    nlp: "spacy.language.Language"

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        stride: int = 0,
    ):
        self.stride = stride
        self.nlp = spacy.load(spacy_model)

    def slice_text(self, doc: Document) -> List[Document]:
        docs = []
        for sent in self.nlp(doc.text).sents:
            if self.stride > 0:
                text = sent.doc[
                    max(sent.start - self.stride, 0) : min(
                        sent.end + self.stride, len(sent.doc)
                    )
                ].text
            else:
                text = sent.text
            docs.append(doc.new(text=str(text)))
        return docs

    def slice_one(self, doc: Document) -> List[Document]:
        if hasattr(doc, "span_groups"):
            raise NotImplementedError(
                "SentenceSlicer does not support slicing documents with "
                "span groups."
            )
        else:
            return self.slice_text(doc)


@slicer_registry.add("block")
class BlockSlicer(SentenceSlicer):
    def __init__(
        self,
        length: int = 300,
        stride: Union[int, float] = 1.0,
        spacy_model: str = "en_core_web_sm",
    ):
        self.length = length

        if isinstance(stride, float):
            assert 0 <= stride <= 1, "overlap must be between 0 and 1"
            stride = int(stride * length)

        super().__init__(spacy_model=spacy_model, stride=stride)

    def slice_text(self, doc: Document) -> List[Document]:
        docs = []
        for i in range(0, len(doc.text), self.stride):
            text = doc.text[i : i + self.length]
            docs.append(doc.new(text=str(text)))
        return docs
