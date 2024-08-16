#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")
source "$PERF_BASE_PATH/logging.sh"

MODEL_SIZE=$1

if [ -z "$MODEL_SIZE" ]; then
    MODEL_SIZE=7
fi

TP_SIZE=$2

if [ -z "$TP_SIZE" ]; then
    TP_SIZE=1
fi

PROMPTS=$3

if [ -z "$PROMPTS" ]; then
    PROMPTS=1000
fi

TURNS=$4

if [ -z "$TURNS" ]; then
    TURNS=1
fi

CLIENTS=$5

if [ -z "$CLIENTS" ]; then
    CLIENTS=1
fi

RAMP_UP_TIME=$6

if [ -z "$RAMP_UP_TIME" ]; then
    RAMP_UP_TIME=1
fi

STOP_TIME=$7

if [ -z "$STOP_TIME" ]; then
    STOP_TIME=300
fi

BACKEND=$8

if [ -z "$BACKEND" ]; then
    BACKEND="vllm"
fi

if [ -z "$HF_MODEL_PATH" ]; then
    echo "[ERROR] Please set HF_MODEL_PATH"
    ERROR "Please set HF_MODEL_PATH"
    exit 1
fi

MODEL_DIR="${HF_MODEL_PATH}/llama-${MODEL_SIZE}b-hf"


if [ -z "$BENCHMARK_LLM" ]; then
    # BENCHMARK_LLM="$PERF_BASE_PATH/python/benchmark_serving_num_clients.py"
    echo "[ERROR] Please set BENCHMARK_LLM"
    ERROR "Please set BENCHMARK_LLM"
    exit 1
fi

if [ -z "$DATASET_PATH" ]; then
    # DATASET_PATH="$PERF_BASE_PATH/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
    # DATASET_PATH="$PERF_BASE_PATH/datasets/samples_1024.json"
    echo "[ERROR] Please set DATASET_PATH"
    ERROR "Please set DATASET_PATH"
    exit 1
fi

if [ -z "$SERVER_URL" ];then
    if [ "$BACKEND" = "vllm" ]; then
        SERVER_URL="${VLLM_SERVER_URL}"
    elif [ "$BACKEND" = "ppl" ]; then
        SERVER_URL="${PPL_SERVER_URL}"
    elif [ "$BACKEND" = "lightllm" ]; then
        SERVER_URL="${LIGHTLLM_SERVER_URL}"
    else
        echo "[ERROR] Please set SERVER_URL"
        ERROR "Please set SERVER_URL"
        exit 1
    fi
fi

CMD="python $BENCHMARK_LLM \
--base-url $SERVER_URL \
--backend $BACKEND \
--model $MODEL_DIR \
--dataset-path $DATASET_PATH \
--num-requests $PROMPTS \
--num-turns $TURNS \
--num-threads $CLIENTS \
--ramp-up-time $RAMP_UP_TIME \
--thread-stop-time $STOP_TIME"

echo "BENCH MODEL${MODEL_SIZE}B TP${TP_SIZE} CLIENTS${CLIENTS} -> $CMD"
INFO "BENCH MODEL${MODEL_SIZE}B TP${TP_SIZE} CLIENTS${CLIENTS} -> $CMD"

eval "$CMD"