#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")
BACKEND_=$BACKEND

function unittest() {
    MODEL_SIZE=$1
    GPUS=$2
    PROMPTS=$3
    TURNS=$4
    CLIENTS=$5
    RAMP_UP_TIME=$6
    STOP_TIME=$7
    MODE=$8

    echo "[BENCHMARK ${MODEL_SIZE} TP${GPUS} TURNS$TURNS CLIENTS$CLIENTS RAMP_UP_TIME$RAMP_UP_TIME STOP_TIME$STOP_TIME ${MODE^^} BACKEND$BACKEND_ DATASET$DATASET]"
    INFO "[BENCHMARK ${MODEL_SIZE} TP${GPUS} TURNS$TURNS CLIENTS$CLIENTS RAMP_UP_TIME$RAMP_UP_TIME STOP_TIME$STOP_TIME ${MODE^^} BACKEND$BACKEND_ DATASET$DATASET]"
    RES=$(bash "$PERF_BASE_PATH/benchmark_one_cuda_${MODE}.sh" "${MODEL_SIZE}" "${GPUS}" "${PROMPTS}" "${TURNS}" "${CLIENTS}" "${RAMP_UP_TIME}" "${STOP_TIME}" "${BACKEND_}" | grep "CSV format output")
    RES=${RES##*:}

    if [ -z "$RES" ]; then
        echo "[FAILED]"
    else
        echo "[OK] $RES"
        INFO "[OK] $RES"
        echo "$MODEL_SIZE,$GPUS,$CLIENTS,$MODE,$RES" >> "$PERF_BASE_PATH/result/benchmark_${BACKEND_}_all_cuda_result.csv"
    fi
}

# function launch_server_and_test() {
#     MODEL_SIZE=$1
#     GPUS=$2
#     PROMPTS=$3
#     RAMP_UP_TIME=$4
#     STOP_TIME=$5
#     MODE=$6

#     SERVER_PID=$(bash "$PERF_BASE_PATH"/benchmark_server_templ_cuda.sh "$MODEL_SIZE" "$GPUS" "$CLIENTS" "$BACKEND" | grep -o "[0-9]\+")
#     SERVER_PID=${SERVER_PID##*:}

#     if [ -z "$SERVER_PID" ]; then
#         echo "[ERROR] SERVER START FAILED"
#         exit 1
#     else
#         echo "SERVER STARTED[$SERVER_PID]: MODEL${MODEL_SIZE}B TP${GPUS}"
#         INFO "SERVER STARTED[$SERVER_PID]: MODEL${MODEL_SIZE}B TP${GPUS}"
#     fi

#     for CLIENTS in "${_NUM_CLIENTS_LIST[@]}"; do
#         for TURNS in "${_NUM_TURNS_LIST[@]}"; do
#             unittest "$MODEL_SIZE" "$GPUS" "$PROMPTS" "$TURNS" "$CLIENTS" "$RAMP_UP_TIME" "$STOP_TIME" "$MODE"
#         done
#     done

#     kill -9 "$SERVER_PID"
#     INFO "SERVER STOPPED[$SERVER_PID]: MODEL${MODEL_SIZE}B TP${GPUS}"
# }


source "$PERF_BASE_PATH/logging.sh"
create_log "$BACKEND"

_NUM_CLIENTS_LIST=(1 5 10 20 30 40 50 100 200 300)
_NUM_TURNS_LIST=(1)

if [ -z "$MODEL_SIZE" ]; then
    echo "set MODEL_SIZE in env_setup.sh first!"
    exit 1
fi

if [ -z "$TP_SIZE" ]; then
    echo "set TP_SIZE in env_setup.sh first!"
    exit 1
fi

if [ -z "$MODE" ]; then
    echo "set MODE in env_setup.sh first!"
    exit 1
fi


for CLIENTS in "${_NUM_CLIENTS_LIST[@]}"; do
for TURNS in "${_NUM_TURNS_LIST[@]}"; do

PROMPTS=1024
# let RAMP_UP_TIME=CLIENTS*0.1
RAMP_UP_TIME=$(echo "scale=2; $CLIENTS*0.1" | bc)
STOP_TIME=10
unittest "$MODEL_SIZE" "$TP_SIZE" "$PROMPTS" "$TURNS" "$CLIENTS" "$RAMP_UP_TIME" "$STOP_TIME" "$MODE"


done
done