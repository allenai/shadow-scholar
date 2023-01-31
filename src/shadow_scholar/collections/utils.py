import boto3
from .cli import get_context

from botocore.client import BaseClient


def get_boto_client(service_name: str, **kwargs) -> BaseClient:
    """Return a boto3 client for the given service name."""

    kwargs = {**get_context().obj.get('boto3_kwargs', {}), **kwargs}
    return boto3.client(service_name, **kwargs)     # pyright: ignore
