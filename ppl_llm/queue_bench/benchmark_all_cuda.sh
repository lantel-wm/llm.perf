SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")

function unittest() {
    MODEL_SIZE=$1
    GPUS=$2
    BATCH=$3
    MODE=$4
    DATASET=$5
    echo "[BENCHMARK ${MODEL_SIZE}B TP${GPUS} BATCH${BATCH} ${MODE^^} ${DATASET^^}]"
    RES=`bash $PERF_BASE_PATH/benchmark_one_cuda_${MODE}.sh ${MODEL_SIZE} ${GPUS} ${BATCH} ${DATASET} | grep "CSV format output"`
    RES=${RES##*:}
    if [ ! -n "$RES" ]; then
        echo "[FAILED]"
    else
        echo "[OK] $RES"
        echo "$MODEL_SIZE,$GPUS,$BATCH,$DATASET,$MODE,$RES" >> $PERF_BASE_PATH/benchmark_all_cuda_result.csv
    fi
}

echo "model_size(B),tp,batch,dataset,mode,generate(ms),prefill(ms),decode(ms),step(ms),prefill_tps,decode_tps,o_tps,io_tps,mem(gib)" > $PERF_BASE_PATH/benchmark_all_cuda_result.csv

_MODE_LIST=(fp16)

_S_BATCH_SIZE_LIST=(4 8 12 16 20 24 28 32 48 64 80 96)
_M_BATCH_SIZE_LIST=(4 8 12 16 20 24 28 32 48 64 80 96 112 128 144 160 176 192)

for MODE in ${_MODE_LIST[@]}; do

for BATCH_SIZE in ${_S_BATCH_SIZE_LIST[@]}; do
    unittest 7 1 $BATCH_SIZE $MODE P99_avg
done
for BATCH_SIZE in ${_S_BATCH_SIZE_LIST[@]}; do
    unittest 7 1 $BATCH_SIZE $MODE P99_prefill
done
for BATCH_SIZE in ${_S_BATCH_SIZE_LIST[@]}; do
    unittest 7 1 $BATCH_SIZE $MODE overall
done

for BATCH_SIZE in ${_M_BATCH_SIZE_LIST[@]}; do
    unittest 70 8 $BATCH_SIZE $MODE P99_avg
done
for BATCH_SIZE in ${_M_BATCH_SIZE_LIST[@]}; do
    unittest 70 8 $BATCH_SIZE $MODE P99_prefill
done
for BATCH_SIZE in ${_M_BATCH_SIZE_LIST[@]}; do
    unittest 70 8 $BATCH_SIZE $MODE overall
done


done
