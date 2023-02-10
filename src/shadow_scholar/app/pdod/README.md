# Paper Details on Demand (PDOD) Demo

This application contains a demo of the Paper Details on Demand (PDOD) system.

You can run this demo in two ways:

- From the command line: use `python -m shadow_scholar app.pdod --`
- As a web application: use `python -m shadow_scholar app.pdod.web --`

## Command Line

The command line application can run in three models:

- Interactive: the user provides documents, and then is prompted to provide
  queries. The results are printed to the console.
- Batch: the user provides a file containing documents and queries. The
  the results are printed to the console or written to a file.
- Evaluation: the user provides files containing documents, queries, and
  qrels. The results are printed to the console or written to a file.
  Metrics are also computed and printed to the console.

Data can be provided in two formats:

- From a registered dataset: by passing
  `python -m shadow_scholar app.pdod -dn <dataset_name>`, the user can use
  a dataset that is registered with the system. Use
  `python -m shadow_scholar -- -h` to see a list of registered datasets.
- From a file: by passing `python -m shadow_scholar app.pdod -df <file_path>`,
  the user can use a file containing documents (or a directory containing
  multiple files). The file must be in JSONL format, and each line must
  contain an object with the following fields: `did` and `text`
    - Queries and qrels can also be provided in this format by using the
        `-qf` and `-rf` flags, respectively. Queries must have fields
        `qid` and `text`, and qrels must have fields `qid`, `did`, and
        `rel`.



## Web Application

The web application is not yet implemented.
