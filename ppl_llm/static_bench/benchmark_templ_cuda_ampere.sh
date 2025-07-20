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
    BATCH=1
fi

IN_LEN=$4

if [ ! -n "$IN_LEN" ]; then
    IN_LEN=8
fi

OUT_LEN=$5

if [ ! -n "$OUT_LEN" ]; then
    OUT_LEN=256
fi

MODEL_DIR="$PERF_BASE_PATH/../opmx_models/llama_${MODEL_SIZE}b_${TP_SIZE}gpu"
MODEL_PARAM_PATH="$PERF_BASE_PATH/../opmx_models/llama_${MODEL_SIZE}b_${TP_SIZE}gpu/params.json"
WARMUP_LOOPS=2
BENCHMARK_LOOPS=2

CMD="${BENCHMARK_LLM} \
--model-dir $MODEL_DIR \
--model-param-path $MODEL_PARAM_PATH \
--tensor-parallel-size $TP_SIZE \
--warmup-loops $WARMUP_LOOPS \
--generation-len $OUT_LEN \
--benchmark-loops $BENCHMARK_LOOPS \
--input-len $IN_LEN \
--batch-size $BATCH \
--micro-batch $BATCH \
$BENCHMARK_EXTENDED_OPTIONS"

echo "BENCH MODEL${MODEL_SIZE}B TP${TP_SIZE} BATCH${BATCH} I${IN_LEN}O${OUT_LEN} -> $CMD"

eval "$CMD"

