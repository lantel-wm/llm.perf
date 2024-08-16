#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")

MODEL_SIZE=$1
TP_SIZE=$2
MODE=$3

source "$PERF_BASE_PATH/logging.sh" $BACKEND
create_server_log

SERVER_PID=$(bash "$PERF_BASE_PATH/start_vllm_server.sh" "$MODEL_SIZE" "$TP_SIZE" "$MODE")
SERVER_URL="${VLLM_SERVER_URL}"


function check_server_status() {
    # change MODEL_DIR according to your model path
    MODEL_DIR="${HF_MODEL_PATH}/llama-${MODEL_SIZE}b-hf"
    CMD="python $PERF_BASE_PATH/python/check_server_status.py --server-url ${SERVER_URL} --backend vllm --model ${MODEL_DIR}"
    status=$(eval "$CMD")

    if [ -z "$status" ]; then
        echo "[ERROR] SERVER STATUS CHECK FAILED"
    fi

    if [ "$status" == "OK" ]; then
        return 0
    else
        return 1
    fi
}


if [ ! -n "$SERVER_PID" ]; then
    echo "[ERROR] SERVER START FAILED"
else
    SERVER_PID=$(echo "$SERVER_PID" | grep -oP "SERVER PID: \K[0-9]+")

    attempt=90
    errno=0
    while [ $attempt -gt 0 ]; do
        # echo "Attempt $attempt"
        sleep 10
        attempt=$((attempt-1))
        check_server_status

        if [ $? -eq 0 ]; then
            break
        fi

        if ! ps -p "$SERVER_PID" > /dev/null; then
            echo "SERVER PID $SERVER_PID is not running"
            ERROR "SERVER PID $SERVER_PID is not running"
            errno=1
            unset SERVER_PID
            break
        fi
    done

    if [ $attempt -eq 0 ]; then
        echo "[ERROR] SERVER START TIMEOUT"
        ERROR "SERVER START TIMEOUT"
    elif [ $errno -eq 1 ]; then
        echo "[ERROR] SERVER START FAILED"
        ERROR "SERVER START FAILED"
    else
        echo "VLLM SERVER STARTED $SERVER_PID"
    fi
fi