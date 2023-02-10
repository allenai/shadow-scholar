import json
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
            "-ep",
            "--dest-path",
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
)
def run_pdod(
    ranker_name: str,
    ranker_kwargs: Optional[dict] = None,
    dataset_name: Optional[str] = None,
    docs_path: Optional[str] = None,
    queries_path: Optional[str] = None,
    qrels_path: Optional[str] = None,
    qrels_mode: Literal["open", "judged"] = "judged",
    dest_path: Optional[str] = None,
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

    if slicer_name:
        slicer = slicer_registry.get(slicer_name)(**(slicer_kwargs or {}))
        dataset.documents = slicer.slice(dataset.documents)

    ranker = ranker_registry.get(ranker_name)(**(ranker_kwargs or {}))
    output: List[List[Document]] = []
    while dataset.queries:
        if len(dataset.queries) == 0 and queries_path is None:
            # interactive mode, ask user for query
            query_text = input("Query: ")
            dataset.queries.append(Query(text=query_text, qid="_"))
        if len(dataset.queries) == 0:
            # no more queries, we're done
            break

        query = dataset.queries.pop(0)

        if len(dataset.qrels) > 0 and qrels_mode == "judged":
            docs = dataset.iter_close(query)
        elif len(dataset.qrels) > 0 and qrels_mode == "open":
            docs = dataset.iter_open(query)
        else:
            docs = dataset.documents

        scored_docs = ranker.score(query.text, docs)
        output.append(scored_docs)

    if dest_path is not None:
        with open(dest_path, "w") as f:
            data = [[doc.as_dict() for doc in docs] for docs in output]
            f.write(json.dumps(data) + "\n")

    if len(dataset.qrels) > 0:
        raise NotImplementedError("Evaluation not yet implemented")
