from argparse import Namespace
import time
from datetime import datetime
from pathlib import Path
from typing import Union, cast

from smashed.utils.io_utils import (
    MultiPath,
    copy_directory,
    remove_directory,
    remove_file,
)
from botocore.client import BaseClient
import boto3

from shadow_scholar.cli import cli, Argument

ATHENA_SQL_DIR = Path(__file__).parent / "athena_sql"


def wait_for_athena_query(
    client, execution_id: str, timeout: int = 5, max_wait: int = 3_600
) -> bool:
    state = "RUNNING"

    print(f"Waiting for {execution_id} to complete..", end='', flush=True)

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
        print(".", end='', flush=True)
        max_wait -= timeout

    raise RuntimeError(f"Query {execution_id} timed out")


def run_athena_query_and_get_result(
    query_string: str,
    s3_staging: Union[str, MultiPath],
    output_location: Union[str, MultiPath],
    output_name: str,
):
    s3_staging = MultiPath.parse(s3_staging)
    output_location = MultiPath.parse(output_location)
    s3_staging = output_location if output_location.is_s3 else s3_staging
    s3_output_location = s3_staging / output_name

    query_string = f"""
        UNLOAD(
            {query_string.rstrip(';')}
        )
        TO '{s3_output_location}'
        WITH (
            format='JSON',
            compression='GZIP'
        );
    """

    athena_client = cast(BaseClient, boto3.client("athena"))
    s3_client = cast(BaseClient, boto3.client("s3"))

    response = athena_client.start_query_execution(
        QueryString=query_string,
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

    print(f"ACL Anthology written to {output_location}")


@cli(
    "collections.s2orc",
    arguments=[
        Argument(
            '-d', '--database',
            default='s2orc_papers',
            type=str,
            help='Athena database name for S2ORC'
        ),
        Argument(
            '-r', '--release',
            default='latest',
            type=str,
            help='S2ORC release name'
        ),
        Argument(
            '-l', '--limit',
            default=10,
            type=int,
            help='Limit number of results; if 0, no limit'
        ),
        Argument(
            '-o', '--output-location',
            required=True,
            type=str,
            help='Location for results; can be an S3 bucket or a local dir'
        ),
        Argument(
            '--s3-staging',
            default='s3://ai2-s2-research/temp/',
            type=str,
            help=(
                'S3 bucket for output of Athena query; to be removed after'
                ' execution if -o/--output-location is a local directory.'
            )
        ),
        Argument(
            '--output-name',
            default=datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
            type=str,
            help=(
                'Name of output directory; by default, '
                'it is the current date/time'
            )
        )
    ]
)
def get_s2orc(
    database: str,
    release: str,
    limit: int,
    output_location: str,
    s3_staging: str,
    output_name: str,
):
    """Get a sample of the S2ORC dataset from Athena."""

    limit_clause = f"LIMIT {limit}" if limit > 0 else ""

    query_string = f"""
        SELECT *
        FROM {database}.{release}
        {limit_clause}
    """

    run_athena_query_and_get_result(
        query_string=query_string,
        s3_staging=s3_staging,
        output_location=output_location,
        output_name=output_name,
    )
