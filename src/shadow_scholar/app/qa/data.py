import gzip
import json
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Manager, Pool, cpu_count
from queue import Queue
from threading import Thread
from typing import Iterable, List, Literal, Optional, Union

from shadow_scholar.cli import Argument, cli, safe_import

with safe_import():
    from smashed.utils.io_utils import (
        MultiPath,
        open_file_for_read,
        open_file_for_write,
        recursively_list_files,
    )
    from tqdm import tqdm


@dataclass
class Section:
    header: Optional[str] = None
    paragraphs: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (f"{self.header}\n\n" if self.header else "") + "\n\n".join(
            self.paragraphs
        )

    def split(self) -> List["Section"]:
        return [
            Section(header=self.header, paragraphs=[p])
            for p in self.paragraphs
        ]

    @property
    def valid(self) -> bool:
        return len(self.paragraphs) > 0


def _process_document(
    doc: dict,
    unit: Union[Literal["paragraph"], Literal["document"]],
) -> Iterable[dict]:
    if (metadata := doc.get("metadata", None)) is None:
        return

    if (title := metadata.get("title", None)) is None:
        return

    if (abstract := metadata.get("abstract", None)) is None:
        return

    if (authors := metadata.get("authors", None)) is None:
        return
    authors = json.loads(authors)

    if (venue := metadata.get("venue", None)) is None:
        return

    if (publication_date := metadata.get("publication_date", None)) is None:
        return
    year = publication_date["year"]

    if (external_ids := metadata.get("external_ids", None)) is None:
        return

    ids = {
        "acl": external_ids.get("acl", None),
        "arxiv": external_ids.get("arxiv", None),
        "doi": external_ids.get("doi", None),
        "corpus": doc["id"],
    }
    tldr = doc["tldr"]

    sections: List[Section] = []

    if doc.get("content", None) is not None:
        # content down here
        raw = (grobid := doc["content"]["grobid"])["contents"]

        heads = json.loads(grobid["annotations"]["section_header"] or "[]")
        paras = json.loads(grobid["annotations"]["paragraph"] or "[]")
        secs_it = (
            (name, loc)
            for name, locs in (("header", heads), ("paragraph", paras))
            for loc in locs
        )

        current_section = Section()

        for name, loc in sorted(secs_it, key=lambda x: x[1]["start"]):
            content = raw[int(loc["start"]) : int(loc["end"])]

            if name == "header":
                if current_section.valid:
                    sections.append(current_section)
                current_section = Section(header=content)
                continue
            current_section.paragraphs.append(content)

        if current_section.paragraphs:
            # add the last section
            sections.append(current_section)

    base = {
        "title": title,
        "authors": authors,
        "venue": venue,
        "year": year,
        "ids": ids,
        "tldr": tldr,
    }

    if unit == "paragraph":
        yield {**base, "paragraph": str(Section("Abstract", [abstract]))}
        for section in sections:
            for paragraph in section.split():
                yield {**base, "paragraph": str(paragraph)}
    elif unit == "document":
        yield {
            **base,
            "abstract": abstract,
            "full_text": [str(sec) for sec in sections],
        }


def _process_collection(
    src: str,
    dst: str,
    unit: Union[Literal["paragraph"], Literal["document"]],
    compressed: bool = True,
    pbar_queue: Optional[Queue] = None,
):
    with open_file_for_read(
        src, "rb" if compressed else "r"
    ) as inf, open_file_for_write(dst, "wb" if compressed else "rb") as outf:
        if compressed:
            inf = gzip.GzipFile(fileobj=inf, mode="rb")
        if compressed:
            outf = gzip.GzipFile(fileobj=outf, mode="wb")

        for ln in inf:
            doc = json.loads(ln)
            for item in _process_document(doc, unit):
                item_str = json.dumps(item) + "\n"
                if compressed:
                    item_str = item_str.encode("utf-8")  # type: ignore
                outf.write(item_str)  # pyright: ignore

            if pbar_queue is not None:
                pbar_queue.put(1)

        outf.close()


def threaded_progressbar(q: Queue, timeout: float):
    with tqdm(desc="Processing...", unit=" doc", unit_scale=True) as pbar:
        while True:
            item = q.get()
            if item is None:
                break
            pbar.update(1)


@cli(
    name="app.qa.collection",
    arguments=[
        Argument(
            "-s",
            "--src",
            type=str,
            required=True,
            help="Path to the collection to process.",
        ),
        Argument(
            "-d",
            "--dst",
            type=str,
            required=True,
            help="Path to the output directory.",
        ),
        Argument(
            "-u",
            "--unit",
            type=str,
            choices=["paragraph", "document"],
            default="paragraph",
            help="Unit to process.",
        ),
        Argument(  # pyright: ignore
            "-D",
            "--Debug",
            dest="debug",
            action="store_true",
            help="Enable debug mode.",
        ),
    ],
    requirements=[
        "smashed[remote]",
    ],
)
def process_collection(
    src: str,
    dst: str,
    unit: Union[Literal["paragraph"], Literal["document"]],
    debug: bool = False,
):
    files = recursively_list_files(src, True)  # pyright: ignore

    if debug:
        for src_fp in files:
            dst_fp = MultiPath.parse(dst) / (src_fp - src)
            _process_collection(src=str(src_fp), dst=str(dst_fp), unit=unit)

    else:
        with Pool(processes=cpu_count()) as pool:
            pbar_queue: Queue = (manager := Manager()).Queue()
            pbar_thread = Thread(
                target=threaded_progressbar,
                args=(pbar_queue, 0.1),
                daemon=True,
            )
            pbar_thread.start()

            results = []
            for src_fp in files:
                dst_fp = MultiPath.parse(dst) / (src_fp - src)
                result = pool.apply_async(
                    partial(
                        _process_collection,
                        src=str(src_fp),
                        dst=str(dst_fp),
                        unit=unit,
                        pbar_queue=pbar_queue,
                    )
                )
                results.append(result)

            pool.close()
            pool.join()

            try:
                while len(results) > 0:
                    results.pop().get()
            except Exception as e:
                raise e
            finally:
                pbar_queue.put(None)
                pbar_thread.join()
                manager.shutdown()
