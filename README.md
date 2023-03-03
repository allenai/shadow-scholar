<h1 align="center">Shadow Scholar</h1>
<p align="center">
    <img src="res/shadow-scholar.png" width="400" height="400" align="center" />
</p>

## Installation

To install, simply clone the repository and install with pip:

```bash
git clone git@github.com:allenai/shadow-scholar.git
cd shadow-scholar
pip install .
```

## Available Scripts

Each command is run with `shadow <command>`.

For a full list of commands, run `shadow -h`.

### Collecting Data

1. `collections.s2orc`: Collects data from the [S2ORC](https://allenai.org/data/s2orc) dataset using an Athena query.

### Applications

1. `app.pdod`: Runs models for the Paper Details On Demand project.

## Getting Access to AWS services

To run the scripts that use AWS services, you will need to have access to the following services:

- [Athena](https://aws.amazon.com/athena/)
- [S3](https://aws.amazon.com/s3/)

The best way to do so is to obtain AWS credentials (access key and secret key) and set them as environment variables.


## Writing your own script

To write your own script for Shadow Scholar, follow these steps:

**Step 1**: Choose where to add your code in Shadow Scholar. It can either be
in an existing module, such as `shadow_scholar.collections.athena`, or in a
new module.

**Step 2**: Understand that the entry point for your script should be a single
function; think of this as the `main` function.

**Step 3**: Write your main function. For each argument you expect a user might
want to provide from command line, add a corresponding argument to the
function. For example:

```python
def my_script(
    arg1: str,
    arg2: int,
    arg3: bool,
    arg4: Optional[str] = None,
):
    # Do something with the arguments
    pass
```

**Step 4**: Add the cli from `shadow_scholar.cli` to your script. This will
allow users to run your script from the command line. For example:

```python
from shadow_scholar.cli import cli
from typing import Optional

@cli(
    name="scripts_collection.my_script",
    arguments=...,
    requirements=...,
)
def my_script(
    arg1: str,
    arg2: int,
    arg3: bool,
    arg4: Optional[str] = None,
):
    # Do something with the arguments
    pass
```

The `cli` decorator takes three arguments: the name of the script, a list
of arguments, and a list of requirements. The name of the script should be
the name a user would use to run the script from the command line. In the
example above, the user would run the script with `shadow
scripts_collection.my_script`.

**Step 5**: Add arguments to your script. Each argument should be an instance
of `shadow_scholar.cli.Argument`. For example:

```python
from typing import Optional
from shadow_scholar.cli import Argument, cli

@cli(
    name="scripts_collection.my_script",
    arguments=[
        Argument(
            name="arg1",
            type=str,
            help="This is the first argument",
        ),
        Argument(
            name="arg2",
            type=int,
            help="This is the second argument",
        ),
        Argument(
            name="arg3",
            type=bool,
            help="This is the third argument",
        ),
        Argument(
            name="arg4",
            type=str,
            help="This is the fourth argument",
            default=None,
        ),
    ],
    requirements=...,
)
def my_script(
    arg1: str,
    arg2: int,
    arg3: bool,
    arg4: Optional[str] = None,
):
    # Do something with the arguments
    pass
```

You should have as many Arguments as you have arguments to your main function.


**Step 6**: Add requirements to your script. Each requirement should be an
in the format used by `requirements.txt`. When using optional requirements,
make sure to wrap them in a `with safe_import()` statement at the top of your
script. For example:

```python
from typing import Optional
from shadow_scholar.cli import Argument, cli, safe_import

with safe_import() as safe:
    # this will not fail if pandas is not installed
    import pandas as pd


@cli(
    name="scripts_collection.my_script",
    arguments=[
        Argument(
            name="arg1",
            type=str,
            help="This is the first argument",
        ),
        Argument(
            name="arg2",
            type=int,
            help="This is the second argument",
        ),
        Argument(
            name="arg3",
            type=bool,
            help="This is the third argument",
        ),
        Argument(
            name="arg4",
            type=str,
            help="This is the fourth argument",
            default=None,
        ),
    ],
    requirements=[
        "pandas>=1.0.0",
    ],
)
def my_script(
    arg1: str,
    arg2: int,
    arg3: bool,
    arg4: Optional[str] = None,
):
    # Do something with the arguments
    pass
```

**Step 7**: Import the function in the `__init__.py` file of the module. For
example, if you added your script to `shadow_scholar/examples.py`, you would
add the following to `shadow_scholar/__init__.py`:

```python
from shadow_scholar.examples import my_script
```
