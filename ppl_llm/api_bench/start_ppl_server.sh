#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")
source "$PERF_BASE_PATH/logging.sh"

if [ -z "$PPL_SERVER_URL" ];then
    PPL_SERVER_URL="127.0.0.1:23333"
    PPL_SERVER_HOST=$(echo $PPL_SERVER_URL | sed -E 's|([^:]+):.*|\1|')
    PPL_SERVER_PORT=$(echo $PPL_SERVER_URL | sed -E 's|.*:([0-9]+)|\1|')
fi

if [ -z "$PPL_SERVER_PATH" ];then
    PPL_SERVER_PATH="/mnt/nvme0n1/workspace/zhaozhiyu/work/llm-bench/install/ppl.llm.serving/ppl-build/ppl_llm_server"
fi

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

# MODEL_DIR="$PERF_BASE_PATH/../../hf_models/llama-${MODEL_SIZE}b-hf"

if [ -z "$OPMX_MODEL_PATH" ]; then
    echo "[ERROR] OPMX_MODEL_PATH is not set"
    exit 1
fi

if [ -z "$HF_MODEL_PATH" ]; then
    echo "[ERROR] HF_MODEL_PATH is not set"
    exit 1
fi

# CHANGE MODEL_DIR ACCORDINGLY
# MODEL_DIR="${OPMX_MODEL_PATH}/llama_${MODEL_SIZE}b_${TP_SIZE}gpu"
MODEL_DIR="${OPMX_MODEL_PATH}/${MODEL_SIZE}B_db1_fq1_fk1_ff1_ac1_qc1_cm0_cl3_${TP_SIZE}gpu"
TOKENIZER_PATH="${HF_MODEL_PATH}/llama-${MODEL_SIZE}b-hf/tokenizer.model"


CMD="NCCL_PROTO=^Simple nohup ${PPL_SERVER_PATH} \
--model-dir ${MODEL_DIR} \
--model-param-path ${MODEL_DIR}/params.json \
--tokenizer-path ${TOKENIZER_PATH} \
--tensor-parallel-size ${TP_SIZE} \
--top-p 0.0 \
--top-k 1 \
--max-tokens-scale 0.94 \
--max-input-tokens-per-request 4096 \
--max-output-tokens-per-request 4096 \
--max-total-tokens-per-request 8192 \
--max-running-batch 1024 \
--max-tokens-per-step 8192 \
--host ${PPL_SERVER_HOST} \
--port ${PPL_SERVER_PORT} \
>> ${PERF_BASE_PATH}/log/server_ppl.log 2>&1 &"

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

