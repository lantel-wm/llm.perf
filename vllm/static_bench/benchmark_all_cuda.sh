#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")

function unittest() {
    MODEL_SIZE=$1
    GPUS=$2
    BATCH=$3
    INLEN=$4
    OUTLEN=$5
    MODE=$6
    echo "[BENCHMARK ${MODEL_SIZE}B TP${GPUS} BATCH${BATCH} I${INLEN} O${OUTLEN} ${MODE^^}]"
    LATENCY=$(bash "$PERF_BASE_PATH/benchmark_one_cuda_${MODE}.sh" "${MODEL_SIZE}" "${GPUS}" "${BATCH}" "${INLEN}" "${OUTLEN}" | grep -oP "Avg latency: \K[0-9]+\.[0-9]+")

    if [ -z "$LATENCY" ]; then
        echo "[FAILED]"
    else
        LATENCY=$(echo "scale=3; $LATENCY * 1000" | bc)
        O_TPS=$(echo "scale=3; $BATCH * $OUTLEN / ($LATENCY / 1000)" | bc)
        IO_TPS=$(echo "scale=3; $BATCH * ($OUTLEN + $INLEN) / ($LATENCY / 1000)" | bc)
        echo "[OK] $LATENCY, $O_TPS, $IO_TPS"
        echo "$MODEL_SIZE,$GPUS,$BATCH,$INLEN,$OUTLEN,$MODE,$LATENCY,$O_TPS,$IO_TPS" >> "$PERF_BASE_PATH/benchmark_all_cuda_result.csv"   
    fi
}

echo "model_size(B),tp,batch,inlen,outlen,mode,latency(ms),o_tps,io_tps" > "$PERF_BASE_PATH/benchmark_all_cuda_result.csv"

_MODE_LIST=(fp16)

_I8O256_BATCH_SIZE_LIST=(1 2 4 8 16 32 48 64 80 96 112 128 160 192 224 256 384 512 768 1024 1280 1536 1792 2048)
_I128O128_BATCH_SIZE_LIST=(1 2 4 8 16 32 48 64 80 96 112 128 160 192 224 256 384 512 768 1024 1280 1536 1792 2048)
_I2048O2048_BATCH_SIZE_LIST=(1 2 4 8 16 32 48 64 80 96 112 128 160 192 224 256)
_I128O2048_BATCH_SIZE_LIST=(1 2 4 8 16 32 48 64 80 96 112 128 160 192 224 256)
_I2048O128_BATCH_SIZE_LIST=(1 2 4 8 16 32 48 64 80 96 112 128 160 192 224 256)
_I1024O1024_BATCH_SIZE_LIST=(1 2 4 8 16 32 48 64 80 96 112 128 160 192 224 256)
_I1024O512_BATCH_SIZE_LIST=(1 2 4 8 16 32 48 64 80 96 112 128 160 192 224 256)
_I512O512_BATCH_SIZE_LIST=(1 2 4 8 16 32 48 64 80 96 112 128 160 192 224 256 384 512)
_I256O512_BATCH_SIZE_LIST=(1 2 4 8 16 32 48 64 80 96 112 128 160 192 224 256 384 512 768 1024)
_I128KO128_BATCH_SIZE_LIST=(1 2 4)
_I64KO128_BATCH_SIZE_LIST=(1 2 4 8)
_I32KO128_BATCH_SIZE_LIST=(1 2 4 8 12 16)
_I16KO128_BATCH_SIZE_LIST=(1 2 4 8 12 16 20 24 28 32)


for MODE in "${_MODE_LIST[@]}"; do


# for BATCH_SIZE in ${_I8O256_BATCH_SIZE_LIST[@]}; do
#     unittest 7 1 $BATCH_SIZE 8 256 $MODE
# done
for BATCH_SIZE in "${_I256O512_BATCH_SIZE_LIST[@]}"; do
    unittest 7 1 "$BATCH_SIZE" 256 512 "$MODE"
done
# for BATCH_SIZE in ${_I512O512_BATCH_SIZE_LIST[@]}; do
#     unittest 7 1 $BATCH_SIZE 512 512 $MODE
# done
# for BATCH_SIZE in ${_I1024O512_BATCH_SIZE_LIST[@]}; do
#     unittest 7 1 $BATCH_SIZE 1024 512 $MODE
# done
# for BATCH_SIZE in ${_I1024O1024_BATCH_SIZE_LIST[@]}; do
#     unittest 7 1 $BATCH_SIZE 1024 1024 $MODE
# done
# for BATCH_SIZE in ${_I16KO128_BATCH_SIZE_LIST[@]}; do
#     unittest 7 1 $BATCH_SIZE 16384 128 $MODE
# done
# for BATCH_SIZE in ${_I32KO128_BATCH_SIZE_LIST[@]}; do
#     unittest 7 1 $BATCH_SIZE 32768 128 $MODE
# done


# for BATCH_SIZE in ${_I8O256_BATCH_SIZE_LIST[@]}; do
#     unittest 13 2 $BATCH_SIZE 8 256 $MODE
# done
# for BATCH_SIZE in ${_I256O512_BATCH_SIZE_LIST[@]}; do
#     unittest 13 2 $BATCH_SIZE 256 512 $MODE
# done
# for BATCH_SIZE in ${_I512O512_BATCH_SIZE_LIST[@]}; do
#     unittest 13 2 $BATCH_SIZE 512 512 $MODE
# done
# for BATCH_SIZE in ${_I1024O512_BATCH_SIZE_LIST[@]}; do
#     unittest 13 2 $BATCH_SIZE 1024 512 $MODE
# done
# for BATCH_SIZE in ${_I1024O1024_BATCH_SIZE_LIST[@]}; do
#     unittest 13 2 $BATCH_SIZE 1024 1024 $MODE
# done
# for BATCH_SIZE in ${_I16KO128_BATCH_SIZE_LIST[@]}; do
#     unittest 13 2 $BATCH_SIZE 16384 128 $MODE
# done
# for BATCH_SIZE in ${_I32KO128_BATCH_SIZE_LIST[@]}; do
#     unittest 13 2 $BATCH_SIZE 32768 128 $MODE
# done


# for BATCH_SIZE in ${_I8O256_BATCH_SIZE_LIST[@]}; do
#     unittest 65 8 $BATCH_SIZE 8 256 $MODE
# done
# for BATCH_SIZE in ${_I256O512_BATCH_SIZE_LIST[@]}; do
#     unittest 65 8 $BATCH_SIZE 256 512 $MODE
# done
# for BATCH_SIZE in ${_I512O512_BATCH_SIZE_LIST[@]}; do
#     unittest 65 8 $BATCH_SIZE 512 512 $MODE
# done
# for BATCH_SIZE in ${_I1024O512_BATCH_SIZE_LIST[@]}; do
#     unittest 65 8 $BATCH_SIZE 1024 512 $MODE
# done
# for BATCH_SIZE in ${_I1024O1024_BATCH_SIZE_LIST[@]}; do
#     unittest 65 8 $BATCH_SIZE 1024 1024 $MODE
# done
# for BATCH_SIZE in ${_I16KO128_BATCH_SIZE_LIST[@]}; do
#     unittest 65 8 $BATCH_SIZE 16384 128 $MODE
# done
# for BATCH_SIZE in ${_I32KO128_BATCH_SIZE_LIST[@]}; do
#     unittest 65 8 $BATCH_SIZE 32768 128 $MODE
# done


# for BATCH_SIZE in ${_I8O256_BATCH_SIZE_LIST[@]}; do
#     unittest 70 8 $BATCH_SIZE 8 256 $MODE
# done
# for BATCH_SIZE in ${_I256O512_BATCH_SIZE_LIST[@]}; do
#     unittest 70 8 $BATCH_SIZE 256 512 $MODE
# done
# for BATCH_SIZE in ${_I512O512_BATCH_SIZE_LIST[@]}; do
#     unittest 70 8 $BATCH_SIZE 512 512 $MODE
# done
# for BATCH_SIZE in ${_I1024O512_BATCH_SIZE_LIST[@]}; do
#     unittest 70 8 $BATCH_SIZE 1024 512 $MODE
# done
# for BATCH_SIZE in ${_I1024O1024_BATCH_SIZE_LIST[@]}; do
#     unittest 70 8 $BATCH_SIZE 1024 1024 $MODE
# done
# for BATCH_SIZE in ${_I16KO128_BATCH_SIZE_LIST[@]}; do
#     unittest 70 8 $BATCH_SIZE 16384 128 $MODE
# done
# for BATCH_SIZE in ${_I32KO128_BATCH_SIZE_LIST[@]}; do
#     unittest 70 8 $BATCH_SIZE 32768 128 $MODE
# done


done
