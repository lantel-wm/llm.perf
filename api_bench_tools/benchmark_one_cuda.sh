#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")

if [ $ENABLE_SYSTEM_PROMPT -eq 0 ]; then
    export BENCHMARK_EXTENDED_OPTIONS=
else
    export BENCHMARK_EXTENDED_OPTIONS="--system-prompt-path $SYSTEM_PROMPT_PATH"
fi

bash "$PERF_BASE_PATH/benchmark_client_templ_cuda.sh" $*