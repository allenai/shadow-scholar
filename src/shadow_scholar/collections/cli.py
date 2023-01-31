import click

CONTEXT_SETTINGS = dict(
    help_option_names=['-h', '--help'],
    default_map={}
)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option(
    '--aws-access-key-id',
    default=None,
    help='AWS access key ID for boto3 client'
)
@click.option(
    '--aws-secret-access-key',
    default=None,
    help='AWS secret access key for boto3 client'
)
@click.option(
    '--aws-region-name',
    default=None,
    help='AWS region name for boto3 client'
)
@click.pass_context
def cli(
    ctx: click.Context,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_region_name: str
):
    """Entry point for the collections subcommand."""
    ctx.ensure_object(dict)

    boto3_kwargs = {
        'aws_access_key_id': aws_access_key_id,
        'aws_secret_access_key': aws_secret_access_key,
        'region_name': aws_region_name,
    }
    ctx.obj.update({'boto3_kwargs': boto3_kwargs})


@click.pass_context
def _get_context(ctx: click.Context):
    return ctx


def get_context():
    """Tries to get the context from the current context stack, otherwise
    creates a new context and returns it."""

    try:
        ctx = _get_context()    # pyright: ignore
    except RuntimeError:
        (ctx := click.Context(cli)).ensure_object(dict)
    return ctx
