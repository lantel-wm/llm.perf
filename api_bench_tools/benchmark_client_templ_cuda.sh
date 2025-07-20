#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")
source "$PERF_BASE_PATH/logging.sh"

MODEL_TAG=$1

if [ -z "$MODEL_TAG" ]; then
    echo "[ERROR] Please set MODEL_TAG"
    ERROR "Please set MODEL_TAG"
    exit 1
fi

PROMPTS=$2

if [ -z "$PROMPTS" ]; then
    PROMPTS=1000
fi

TURNS=$3

if [ -z "$TURNS" ]; then
    TURNS=1
fi

CLIENTS=$4

if [ -z "$CLIENTS" ]; then
    CLIENTS=1
fi

RAMP_UP_TIME=$5

if [ -z "$RAMP_UP_TIME" ]; then
    RAMP_UP_TIME=1
fi

STOP_TIME=$6

if [ -z "$STOP_TIME" ]; then
    STOP_TIME=300
fi

BACKEND=$7

if [ -z "$BACKEND" ]; then
    BACKEND="vllm"
fi

if [ -z "$HF_MODEL_PATH" ]; then
    echo "[ERROR] Please set HF_MODEL_PATH"
    ERROR "Please set HF_MODEL_PATH"
    exit 1
fi

if [ -z "$DATASET" ]; then
    DATASET="sharegpt"
fi

if [ -z "$BENCHMARK_LLM" ]; then
    echo "[ERROR] Please set BENCHMARK_LLM"
    ERROR "Please set BENCHMARK_LLM"
    exit 1
fi


if [ -z "$SERVER_URL" ];then
    if [ "$BACKEND" = "vllm" ]; then
        SERVER_URL="${VLLM_SERVER_URL}"
    elif [ "$BACKEND" = "ppl" ]; then
        SERVER_URL="${PPL_SERVER_URL}"
    elif [ "$BACKEND" = "lightllm" ]; then
        SERVER_URL="${LIGHTLLM_SERVER_URL}"
    elif [ "$BACKEND" = "amsv2" ]; then
        SERVER_URL="${AMSV2_SERVER_URL}"
    elif [ "$BACKEND" = "sglang" ]; then
        SERVER_URL="${SGLANG_SERVER_URL}"
    else
        echo "[ERROR] Unsupported backend $BACKEND"
        ERROR "Unsupported backend $BACKEND"
        exit 1
    fi
fi

if [ -z "$LOG_LEVEL" ];then
    LOG_LEVEL="WARNING"
fi


if [ "$DATASET" = "sharegpt" ]; then
    DATASET_PATH="$SHAREGPT_DATASET_PATH"
elif [ "$DATASET" = "xiaomi" ]; then
    DATASET_PATH="$XIAOMI_DATASET_PATH"
else
    echo "[ERROR] unknown dataset $DATASET"
fi

INFO "Using $DATASET dataset, dataset path: $DATASET_PATH"

CMD="python $BENCHMARK_LLM \
--base-url $SERVER_URL \
--backend $BACKEND \
--model $MODEL_TAG \
--tokenizer $BENCHMARK_TOKENIZER_PATH \
--dataset $DATASET \
--dataset-path $DATASET_PATH \
--num-requests $PROMPTS \
--num-turns $TURNS \
--num-clients $CLIENTS \
--ramp-up-time $RAMP_UP_TIME \
--stop-time $STOP_TIME \
--log-file $PERF_BASE_PATH/log/benchmark_all_cuda.log \
--log-level $LOG_LEVEL \
--execute-mode $CLIENT_EXECUTE_MODE \
$BENCHMARK_EXTENDED_OPTIONS"

echo "BENCH MODEL${MODEL_TAG} TP${TP_SIZE} CLIENTS${CLIENTS} -> $CMD"
INFO "BENCH MODEL${MODEL_TAG} TP${TP_SIZE} CLIENTS${CLIENTS} -> $CMD"

eval "$CMD"