#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")

source "$PERF_BASE_PATH/benchmark_config.sh"
BACKEND_=$BACKEND

source "$PERF_BASE_PATH/logging.sh"
create_log "$BACKEND"

_NUM_CLIENTS_LIST=($NUM_CLIENTS)
_NUM_TURNS_LIST=(1)

if [ -z "$MODEL_TAG" ]; then
    echo "set MODEL_TAG in benchmark_config.sh first!"
    exit 1
fi

INFO "MODEL_TAG: $MODEL_TAG"
echo "MODEL_TAG: $MODEL_TAG"

INFO "BACKEND: $BACKEND_"
echo "BACKEND: $BACKEND_"


function unittest() {
    MODEL_TAG=$1
    PROMPTS=$2
    TURNS=$3
    CLIENTS=$4
    RAMP_UP_TIME=$5
    STOP_TIME=$6
    MODE=$7

    echo "[BENCHMARK ${MODEL_TAG} TURNS$TURNS CLIENTS$CLIENTS RAMP_UP_TIME$RAMP_UP_TIME STOP_TIME$STOP_TIME BACKEND$BACKEND_ DATASET$DATASET]"
    INFO "[BENCHMARK ${MODEL_TAG} TURNS$TURNS CLIENTS$CLIENTS RAMP_UP_TIME$RAMP_UP_TIME STOP_TIME$STOP_TIME BACKEND$BACKEND_ DATASET$DATASET]"
    RES=$(bash "$PERF_BASE_PATH/benchmark_one_cuda.sh" "${MODEL_TAG}" "${PROMPTS}" "${TURNS}" "${CLIENTS}" "${RAMP_UP_TIME}" "${STOP_TIME}" "${BACKEND_}" | grep "CSV format output")
    RES=${RES##*:}

    if [ -z "$RES" ]; then
        echo "[FAILED]"
    else
        echo "[OK] $RES"
        INFO "[OK] $RES"
        echo "$MODEL_TAG,$CLIENTS,$RES" >> "$BENCHMARK_RESULT_PATH"
    fi
}


for CLIENTS in "${_NUM_CLIENTS_LIST[@]}"; do
for TURNS in "${_NUM_TURNS_LIST[@]}"; do

PROMPTS=1024
# RAMP_UP_TIME=CLIENTS*0.1
RAMP_UP_TIME=$(echo "scale=2; $CLIENTS*0.1" | bc)
_STOP_TIME=$STOP_TIME
unittest "$MODEL_TAG" "$PROMPTS" "$TURNS" "$CLIENTS" "$RAMP_UP_TIME" "$_STOP_TIME" "$MODE"


done
done