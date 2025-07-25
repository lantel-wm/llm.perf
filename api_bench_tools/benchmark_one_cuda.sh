#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")

if [ $ENABLE_SYSTEM_PROMPT -eq 1 ]; then
    export BENCHMARK_EXTENDED_OPTIONS="$BENCHMARK_EXTENDED_OPTIONS --system-prompt-path $SYSTEM_PROMPT_PATH"
else
    export BENCHMARK_EXTENDED_OPTIONS=$BENCHMARK_EXTENDED_OPTIONS
fi

if [ $ALLOW_REPETITIVE_REQUESTS -eq 1 ]; then
    export BENCHMARK_EXTENDED_OPTIONS="$BENCHMARK_EXTENDED_OPTIONS --allow-repetitive-requests"
else
    export BENCHMARK_EXTENDED_OPTIONS=$BENCHMARK_EXTENDED_OPTIONS
fi

bash "$PERF_BASE_PATH/benchmark_client_templ_cuda.sh" $*