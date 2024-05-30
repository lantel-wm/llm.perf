SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")

export BENCHMARK_EXTENDED_OPTIONS="--quant-method online_i8i8"

bash $PERF_BASE_PATH/benchmark_templ_cuda.sh $*
