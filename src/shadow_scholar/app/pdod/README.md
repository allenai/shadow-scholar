# Paper Details on Demand (PDOD) Demo

This application contains a demo of the Paper Details on Demand (PDOD) system.

You can run this demo in two ways:

- From the command line: use `shadow app.pdod ...`
- As a web application: use `shadow app.pdod.web ...`

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
  `shadow app.pdod -dn <dataset_name>`, the user can use a dataset that is
  registered with the system. Use `shadow -h` to see a list of registered
  datasets.
- From a file: by passing `shadow app.pdod -df <file_path>`,
  the user can use a file containing documents (or a directory containing
  multiple files). The file must be in JSONL format, and each line must
  contain an object with the following fields: `did` and `text`
    - Queries and qrels can also be provided in this format by using the
      `-qf` and `-rf` flags, respectively. Queries must have fields
      qid` and `text`, and qrels must have fields `qid`, `did`, and
      `rel`.


### Examples

Slice documents sentence by sentence and rank them using TF-IDF. Query will
be supplied interactively.


```bash
shadow app.pdod \
    -dp "src/shadow_scholar/app/pdod/examples/docs.jsonl" \
    -rn "tfidf" \
    -sl "sent"
```

Slice documents in blocks of 64 tokens with a 32 token overlap and rank them
using dense embeddings (encoded using Contriver). Queries will be supplied
interactively.

```bash
shadow app.pdod \
    -dp "src/shadow_scholar/app/pdod/examples/docs.jsonl" \
    -rn "dense" \
    -rk '{"model_name_or_path": "facebook/contriever"}' \
    -sl "block" \
    -sk '{"length": 32, "stride": 0.5}'
```

Slice documents by sentence, rank them using dense embeddings (encoded using
Contriver), and run for queries at the given path. Results will be written
to an output file.

```bash
shadow app.pdod \
    -dp "src/shadow_scholar/app/pdod/examples/docs.jsonl" \
    -rn "dense" \
    -rk '{"model_name_or_path": "facebook/contriever"}' \
    -sl "sent" \
    -qp "src/shadow_scholar/app/pdod/examples/queries.jsonl" \
    -op "src/shadow_scholar/app/pdod/examples/results_dense_sent.jsonl"
```

## Web Application

The web application is not yet implemented.
