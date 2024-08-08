#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")
source "$PERF_BASE_PATH/logging.sh"

if [ -z "$LIGHTLLM_SERVER_URL" ];then
    echo "[ERROR] LIGHTLLM_SERVER_URL is not set"
    exit 1
fi

LIGHTLLM_SERVER_HOST=$(echo $LIGHTLLM_SERVER_URL | sed -E 's|http://([^:/]+).*|\1|')
LIGHTLLM_SERVER_PORT=$(echo $LIGHTLLM_SERVER_URL | sed -E 's|.*:([0-9]+)|\1|')

MODEL_SIZE=$1

if [ -z "$MODEL_SIZE" ]; then
    MODEL_SIZE=7
fi

TP_SIZE=$2

if [ -z "$TP_SIZE" ]; then
    TP_SIZE=1
fi

MODEL=$3

if [ -z "$MODEL" ]; then
    MODEL="fp16"
fi

if [ -z "$HF_MODEL_PATH" ]; then
    # HF_MODEL_PATH="/mnt/llm/llm_perf/hf_models"
    echo "[ERROR] HF_MODEL_PATH is not set"
    exit 1
fi

# CHANGE MODEL_DIR ACCORDINGLY
MODEL_DIR="${HF_MODEL_PATH}/llama-${MODEL_SIZE}b-hf"

# python -m vllm.entrypoints.openai.api_server --model /mnt/llm2/llm_perf/hf_models/llama-7b-hf --swap-space 16 --disable-log-requests --enforce-eager --host 10.198.31.25  --port 8000

CMD="nohup python -m lightllm.server.api_server \
--model_dir $MODEL_DIR \
--host $LIGHTLLM_SERVER_HOST \
--port $LIGHTLLM_SERVER_PORT \
--tp $TP_SIZE \
--max_total_token_num 120000 \
--tokenizer_mode fast \
>> ${PERF_BASE_PATH}/log/server_lightllm.log 2>&1 &"

# --disable-log-stats \

echo "SERVER STARTING: MODEL${MODEL_SIZE}B TP${TP_SIZE} HOST${HOST} PORT${PORT} -> $CMD"
INFO "SERVER STARTING: MODEL${MODEL_SIZE}B TP${TP_SIZE} HOST${HOST} PORT${PORT} -> $CMD"

eval "$CMD"

SERVER_PID=$!

if [ -z "$SERVER_PID" ]; then
    echo "[ERROR] SERVER START FAILED"
    ERROR "SERVER START FAILED"
else
    echo "SERVER PID: $SERVER_PID"
    INFO "SERVER PID: $SERVER_PID"
fi
