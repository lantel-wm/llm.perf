SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")

if [ ! -n "$BENCHMARK_LLM" ]; then
    echo "[ERROR] please set env BENCHMARK_LLM to the benchmark_llama executable"
    exit 1
fi

MODEL_SIZE=$1

if [ ! -n "$MODEL_SIZE" ]; then
    MODEL_SIZE=7
fi

TP_SIZE=$2

if [ ! -n "$TP_SIZE" ]; then
    TP_SIZE=1
fi

BATCH=$3

if [ ! -n "$BATCH" ]; then
    BATCH=4
fi

DATASET=$4
if [ ! -n "$DATASET" ]; then
    DATASET="overall"
fi

IN_IDS_FILE="$PERF_BASE_PATH/dataset/${DATASET}_input_token_ids.txt"
OUT_LEN_FILE="$PERF_BASE_PATH/dataset/${DATASET}_output_length.txt"

MODEL_DIR="$PERF_BASE_PATH/../opmx_models/llama_${MODEL_SIZE}b_${TP_SIZE}gpu"
MODEL_PARAM_PATH="$PERF_BASE_PATH/../opmx_models/llama_${MODEL_SIZE}b_${TP_SIZE}gpu/params.json"
WARMUP_LOOPS=2
BENCHMARK_LOOPS=1

CMD="NCCL_PROTO=^Simple ${BENCHMARK_LLM} \
--model-dir $MODEL_DIR \
--model-param-path $MODEL_PARAM_PATH \
--tensor-parallel-size $TP_SIZE \
--warmup-loops $WARMUP_LOOPS \
--benchmark-loops $BENCHMARK_LOOPS \
--batch-size 1000 \
--micro-batch $BATCH \
--input-ids-file $IN_IDS_FILE \
--generation-lens-file $OUT_LEN_FILE \
$BENCHMARK_EXTENDED_OPTIONS"

echo "BENCH MODEL${MODEL_SIZE}B TP${TP_SIZE} BATCH${BATCH} -> $CMD"

eval "$CMD"

