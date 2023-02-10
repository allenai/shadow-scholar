import json
import os
from typing import List, Literal, Optional

from shadow_scholar.cli import Argument, cli, load_kwargs

from .datasets import Document, Qrel, Query, dataset_registry
from .rankers import ranker_registry
from .slicers import slicer_registry


@cli(
    name="app.pdod",
    arguments=[
        Argument(
            "-rn",
            "--ranker-name",
            help=f"Ranker to use. Options: {ranker_registry.keys()}",
            required=True,
        ),
        Argument(
            "-rk",
            "--ranker-kwargs",
            help="JSON-encoded kwargs for the ranker",
            type=load_kwargs,
        ),
        Argument(
            "-dn",
            "--dataset-name",
            help=f"Dataset to use. Options: {dataset_registry.keys()}",
        ),
        Argument(
            "-dp",
            "--docs-path",
            help=(
                "Path to file or directory containing a JSONL of documents. "
                f"Documents are expect to have fields: {Document.fields()}"
            ),
        ),
        Argument(
            "-qp",
            "--queries-path",
            help=(
                "Path to file or directory containing a JSONL of queries. "
                f"Queries are expect to have fields: {Query.fields()}"
            ),
        ),
        Argument(
            "-rp",
            "--qrels-path",
            help=(
                "Path to file or directory containing a JSONL of qrels. "
                f"Qrels are expect to have fields: {Qrel.fields()}"
            ),
        ),
        Argument(
            "-qm",
            "--qrels-mode",
            help="Whether to use open or judged qrels when qrels are provided",
            choices=["open", "judged"],
            default="judged",
        ),
        Argument(
            "-op",
            "--output-path",
            help="If provided, path to write results to",
        ),
        Argument(
            "-sl",
            "--slicer-name",
            help=(
                f"Slicer to use. Options: {slicer_registry.keys()} "
                "If not provided, no slicing will be performed."
            ),
        ),
        Argument(
            "-sk",
            "--slicer-kwargs",
            help="JSON-encoded kwargs for the slicer",
            type=load_kwargs,
        ),
    ],
    requirements=[
        "numpy",
        "spacy",
        "torch",
        "scikit-learn",
        "transformers",
        "unidecode",
        "sentence-transformers",
    ],
)
def run_pdod(
    ranker_name: str,
    ranker_kwargs: Optional[dict] = None,
    dataset_name: Optional[str] = None,
    docs_path: Optional[str] = None,
    queries_path: Optional[str] = None,
    qrels_path: Optional[str] = None,
    qrels_mode: Literal["open", "judged"] = "judged",
    output_path: Optional[str] = None,
    slicer_name: Optional[str] = None,
    slicer_kwargs: Optional[dict] = None,
):
    if docs_path:
        dataset = dataset_registry.get("from_files")(
            docs_path=docs_path,
            queries_path=queries_path,
            qrels_path=qrels_path,
        )
    elif dataset_name:
        dataset = dataset_registry.get(dataset_name)()
    else:
        raise ValueError("Either dataset_name or docs_path must be provided")

    is_interactive = len(dataset.queries) == 0 and queries_path is None
    is_evaluation = len(dataset.qrels) > 0

    if slicer_name:
        slicer = slicer_registry.get(slicer_name)(**(slicer_kwargs or {}))
        dataset.documents = slicer.slice(dataset.documents)

    ranker = ranker_registry.get(ranker_name)(**(ranker_kwargs or {}))
    ranker.warmup(dataset.documents)
    output: List[Document] = []

    while True:
        if is_interactive:
            # interactive mode, ask user for query
            try:
                query_text = input("Query: ")
            except KeyboardInterrupt:
                print("Exiting...")
                break
            dataset.queries.append(Query(text=query_text, qid="_"))
        if len(dataset.queries) == 0:
            # no more queries, we're done
            break

        # take the next query
        query = dataset.queries.pop(0)

        if is_evaluation and qrels_mode == "judged":
            docs = dataset.iter_close(query)
        elif is_evaluation and qrels_mode == "open":
            docs = dataset.iter_open(query)
        else:
            docs = dataset.documents

        scored_docs = ranker.score(query.text, docs)

        if is_interactive:
            width = os.get_terminal_size().columns
            # only show the top 5 results
            for doc in scored_docs[:5]:
                print(f"{doc.score:.3f} {doc.text[:width - 10]}...")
            print()
        else:
            scored_docs = [doc.query(query.qid) for doc in scored_docs]

        output.extend(scored_docs)

    if output_path is not None:
        with open(output_path, "w") as f:
            for doc in output:
                f.write(json.dumps(doc.as_dict()) + "\n")

    if len(dataset.qrels) > 0:
        raise NotImplementedError("Evaluation not yet implemented")


@cli(name="app.pdod.web")
def run_pdod_web_ui():
    raise NotImplementedError("Web UI not yet implemented")
