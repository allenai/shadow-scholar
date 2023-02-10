#! /usr/bin/env bash

# path to directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# running the app
python -m shadow_scholar app.pdod \
    -dp "${DIR}/docs.jsonl" \
    -rn "tfidf" \
    -sl "sent"
