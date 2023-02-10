import time
from datetime import datetime
from pathlib import Path
from string import Formatter
from typing import Any, Dict, Optional, Union, cast

from shadow_scholar.cli import Argument, cli, safe_import

with safe_import():
    import boto3
    from botocore.client import BaseClient
    from smashed.utils.io_utils import (
        MultiPath,
        copy_directory,
        remove_directory,
        remove_file,
    )


QUERIES_DIR = Path(__file__).parent / "queries"


def wait_for_athena_query(
    client, execution_id: str, timeout: int = 5, max_wait: int = 3_600
) -> bool:
    state = "RUNNING"

    print(f"Waiting for {execution_id} to complete..", end="", flush=True)

    while max_wait > 0 and state in ["RUNNING", "QUEUED"]:
        response = client.get_query_execution(QueryExecutionId=execution_id)
        if (
            "QueryExecution" in response
            and "Status" in response["QueryExecution"]
            and "State" in response["QueryExecution"]["Status"]
        ):
            state = response["QueryExecution"]["Status"]["State"]
            if state == "SUCCEEDED":
                print(f"\nQuery {execution_id} succeeded!", flush=True)
                return True
            elif state == "FAILED":
                err = response["QueryExecution"]["Status"]["StateChangeReason"]
                raise RuntimeError(f"Query {execution_id} failed: {err}")

        time.sleep(timeout)
        print(".", end="", flush=True)
        max_wait -= timeout

    raise RuntimeError(f"Query {execution_id} timed out")


@cli(
    name="collections.run_athena_query",
    arguments=[
        Argument(
            "-q",
            "--query-path",
            type=str,
            required=True,
            help="Path to the sql query file to run.",
        ),
        Argument(
            "--s3-staging",
            default="s3://ai2-s2-research/temp/",
            type=str,
            help=(
                "S3 bucket for output of Athena query; to be removed after"
                " execution if -o/--output-location is a local directory."
            ),
        ),
        Argument(
            "--output-name",
            default=datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
            type=str,
            help="Name of output directory; by default, current date/time",
        ),
    ],
    requirements=["boto3", "botocore", "smashed[remote]"],
)
def run_athena_query_and_get_result(
    query_path: str,
    s3_staging: Union[str, "MultiPath"],
    output_location: Union[str, "MultiPath"],
    output_name: str,
    template_args: Optional[Dict[str, Any]] = None,
):
    s3_staging = MultiPath.parse(s3_staging)
    output_location = MultiPath.parse(output_location)
    s3_staging = output_location if output_location.is_s3 else s3_staging
    s3_output_location = s3_staging / output_name
    template_args = template_args or {}

    with open(query_path) as f:
        query_string = f.read()

    formatting_keys = [
        t[1] for t in Formatter().parse(query_string) if t[1] is not None
    ]
    formatting_data = {
        **{k: "" for k in formatting_keys},
        **(template_args or {}),
    }

    formatted_query_string = query_string.format(**formatting_data)

    athena_client = cast(BaseClient, boto3.client("athena"))
    s3_client = cast(BaseClient, boto3.client("s3"))

    response = athena_client.start_query_execution(
        QueryString=formatted_query_string,
        ResultConfiguration=dict(OutputLocation=s3_staging.as_str),
    )
    execution_id = response["QueryExecutionId"]
    wait_for_athena_query(athena_client, response["QueryExecutionId"])

    # remove metadata and execution manifest from temporary S3 bucket
    manifest_loc = s3_staging / f"{execution_id}-manifest.csv"
    remove_file(manifest_loc, client=s3_client)

    metadata_loc = s3_staging / f"{execution_id}.metadata"
    remove_file(metadata_loc, client=s3_client)

    if not output_location.is_s3:
        copy_directory(s3_output_location, output_location)
        remove_directory(s3_output_location, client=s3_client)

    print(f"Dataset written to {output_location}")


@cli(
    name="collections.s2ag.abstracts",
    arguments=[
        Argument(
            "-l",
            "--limit",
            default=10,
            type=int,
            help="Limit number of results; if 0, no limit",
        ),
        Argument(
            "-a",
            "--abstract",
            default=None,
            type=str,
            help="text to search in the abstract",
        ),
        *run_athena_query_and_get_result.args,
    ],
    requirements=run_athena_query_and_get_result.reqs,
)
def get_s2ag_abstracts(
    database: str,
    release: str,
    limit: int,
    output_location: str,
    s3_staging: str,
    output_name: str,
    query_path: str = str(QUERIES_DIR / "s2ag_abstracts.sql"),
):
    """Get a sample of the S2ORC dataset from Athena."""

    template_args = {
        "limit_clause": f"LIMIT {limit}" if limit > 0 else "",
    }

    run_athena_query_and_get_result(
        query_path=query_path,
        s3_staging=s3_staging,
        output_location=output_location,
        output_name=output_name,
        template_args=template_args,
    )
