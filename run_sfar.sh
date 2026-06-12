#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU="0"
RUN_NAME="full_graph_gcn"

if [[ $# -gt 0 && "$1" != --* ]]; then
  GPU="$1"
  shift
fi

if [[ $# -gt 0 && "$1" != --* ]]; then
  RUN_NAME="$1"
  shift
fi

python main.py \
  --gpu "$GPU" \
  --gcn-graph-scope full \
  --run-name "$RUN_NAME" \
  "$@"
