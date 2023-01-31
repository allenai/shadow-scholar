from pathlib import Path

import click

from .cli import cli
from .utils import get_boto_client

ATHENA_SQL_DIR = Path(__file__).parent / "athena_sql"


@cli.command('acl')
@click.option(
    "-q",
    "--query-sql-path",
    default=ATHENA_SQL_DIR / "acl.sql",
    type=click.Path(exists=True),
    help='Path to SQL file containing the query to run on Athena',
)
@click.option(
    '-d',
    '--database',
    default='s2orc_papers',
    help='Athena database name for S2ORC'
)
def get_acl_anthology(
    query_sql_path: Path,
    database: str
):
    cl = get_boto_client('athena')
    print(cl)
    # client = boto3.client(
    #     "athena",
    #     aws_access_key_id=ctx.obj["aws_access_key_id"],
    #     aws_secret_access_key=ctx.obj["aws_secret_access_key"],
    #     region_name=ctx.obj["aws_region"],
    #     aws_profile=ctx.obj["aws_profile"],
    # )

    # with open(query_sql_path) as f:
    #     query_sql = f.read()

    # response = client.start_query_execution(
    #     QueryString=query_sql
