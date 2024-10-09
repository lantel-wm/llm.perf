#!/bin/bash

BACKEND_=$BACKEND

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")
LOG_DIR="$PERF_BASE_PATH/log/benchmark_all_cuda.log"
SERVER_LOG_DIR="$PERF_BASE_PATH/log/server_$BACKEND_.log"

if [ ! -d "$PERF_BASE_PATH/log" ]; then
    mkdir -p "$PERF_BASE_PATH/log"
fi

if [ ! -d "$PERF_BASE_PATH/result" ]; then
    mkdir -p "$PERF_BASE_PATH/result"
fi

function create_server_log() {
    if [ -f "$SERVER_LOG_DIR" ]; then
        mv "$SERVER_LOG_DIR" "$PERF_BASE_PATH/log/server_${BACKEND_}_${log_date}.log"
    fi

    touch "$SERVER_LOG_DIR"
}

function create_log() {
    if [ -f "$LOG_DIR" ]; then
        log_date=$(head -n 1 "$LOG_DIR" | grep -oP "\d+")
        mv "$LOG_DIR" "$PERF_BASE_PATH/log/benchmark_all_cuda_${log_date}.log"
    fi

    
    if [ -f "$PERF_BASE_PATH/result/benchmark_${BACKEND_}_all_cuda_result.csv" ]; then
        mkdir -p "$PERF_BASE_PATH/result/${log_date}"
        mv "$PERF_BASE_PATH/result/benchmark_${BACKEND_}_all_cuda_result.csv" "$PERF_BASE_PATH/result/${log_date}"
        mv "$PERF_BASE_PATH/result/env_setup.sh" "$PERF_BASE_PATH/result/${log_date}"
        cp "$PERF_BASE_PATH/log/benchmark_all_cuda_${log_date}.log" "$PERF_BASE_PATH/result/${log_date}"
    fi

    echo "[INFO] benchmark_all_cuda.sh started at $(date +"%Y%m%d%H%M%S")" > "$LOG_DIR"

    echo "model_tag,num_clients,completed,success_rate,qps,total_inlen,total_outlen,avg_inlen,avg_outlen,max_inlen,max_outlen,o_tps,io_tps,\
min_ttft,max_ttft,mean_ttft,median_ttft,std_ttft,p90_ttft,p99_ttft,\
min_tpot,max_tpot,mean_tpot,median_tpot,std_tpot,p90_tpot,p99_tpot,\
min_e2e,max_e2e,mean_e2e,median_e2e,std_e2e,p90_e2e,p99_e2e,\
min_itl,max_itl,mean_itl,median_itl,std_itl,p90_itl,p99_itl" > "$PERF_BASE_PATH/result/benchmark_${BACKEND_}_all_cuda_result.csv"
    cp "$PERF_BASE_PATH/env_setup.sh" "$PERF_BASE_PATH/result"
}

function INFO() {
    if [ -f "$LOG_DIR" ]; then
        echo "[INFO] $(date +"%Y-%m-%d %H:%M:%S") $1" >> "$LOG_DIR"
    fi
}

function WARNING() {
    if [ -f "$LOG_DIR" ]; then
        echo "[WARNING] $(date +"%Y-%m-%d %H:%M:%S") $1" >> "$LOG_DIR"
    fi
}

function ERROR() {
    if [ -f "$LOG_DIR" ]; then
        echo "[ERROR] $(date +"%Y-%m-%d %H:%M:%S") $1" >> "$LOG_DIR"
    fi
}

function FATAL() {
    if [ -f "$LOG_DIR" ]; then
        echo "[FATAL] $(date +"%Y-%m-%d %H:%M:%S") $1" >> "$LOG_DIR"
    fi
}