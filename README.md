<h1 align="center">Shadow Scholar</h1>
<p align="center">
    <img src="res/shadow-scholar.png" width="400" height="400" align="center" />
</p>

## Installation

To install, simply clone the repository and install with pip:

```bash
git clone git@github.com:allenai/shadow-scholar.git
cd shadow-scholar
pip install -e .
```

## Available Scripts

Each command is run with `python -m shadow_scholar <command>`.

For a full list of commands, run `python -m shadow_scholar -h`.

### Collecting Data

1. `collections.s2orc`: Collects data from the [S2ORC](https://allenai.org/data/s2orc) dataset using an Athena query.

## Getting Access to AWS services

To run the scripts that use AWS services, you will need to have access to the following services:

- [Athena](https://aws.amazon.com/athena/)
- [S3](https://aws.amazon.com/s3/)

The best way to do so is to obtain AWS credentials (access key and secret key) and set them as environment variables.

Alternatively, you can pass the credentials as command line arguments.  For example, to run the `s2orc` command, you can run:

```bash
python -m shadow_scholar \
    --aws-access-key-id <access-key-id> \
    --aws-secret-access-key <secret-access-key> \
    collections.s2orc
```
