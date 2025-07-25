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
    RES=`bash $PERF_BASE_PATH/benchmark_one_cuda_${MODE}.sh ${MODEL_SIZE} ${GPUS} ${BATCH} ${INLEN} ${OUTLEN} | grep "CSV format output"`
    RES=${RES##*:}
    if [ ! -n "$RES" ]; then
        echo "[FAILED]"
    else
        echo "[OK] $RES"
        echo "$MODEL_SIZE,$GPUS,$BATCH,$INLEN,$OUTLEN,$MODE,$RES" >> $PERF_BASE_PATH/benchmark_all_cuda_result.csv        
    fi
}

function unittest_loop() {
    MODEL_SIZE=$1
    GPUS=$2
    local -n BATCH_LIST=$3
    INLEN=$4
    OUTLEN=$5
    MODE=$6
    for BATCH_SIZE in ${BATCH_LIST[@]}; do
        unittest $MODEL_SIZE $GPUS $BATCH_SIZE $INLEN $OUTLEN $MODE
        if [ $? -ne 0 ]; then
            echo "[INFO] break at batch $BATCH_SIZE"
            break
        fi
    done
}

# example: unittest_loop 7 1 _I256O512_BATCH_SIZE_LIST 256 512 $MODE

echo "model_size(B),tp,batch,inlen,outlen,mode,generate(ms),prefill(ms),decode(ms),step(ms),prefill_tps,decode_tps,o_tps,io_tps,mem(gib)" > $PERF_BASE_PATH/benchmark_all_cuda_result.csv

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


for MODE in ${_MODE_LIST[@]}; do


for BATCH_SIZE in ${_I8O256_BATCH_SIZE_LIST[@]}; do
    unittest 7 1 $BATCH_SIZE 8 256 $MODE
done
for BATCH_SIZE in ${_I256O512_BATCH_SIZE_LIST[@]}; do
    unittest 7 1 $BATCH_SIZE 256 512 $MODE
done
for BATCH_SIZE in ${_I512O512_BATCH_SIZE_LIST[@]}; do
    unittest 7 1 $BATCH_SIZE 512 512 $MODE
done
for BATCH_SIZE in ${_I1024O512_BATCH_SIZE_LIST[@]}; do
    unittest 7 1 $BATCH_SIZE 1024 512 $MODE
done
for BATCH_SIZE in ${_I1024O1024_BATCH_SIZE_LIST[@]}; do
    unittest 7 1 $BATCH_SIZE 1024 1024 $MODE
done
for BATCH_SIZE in ${_I128O128_BATCH_SIZE_LIST[@]}; do
    unittest 7 1 $BATCH_SIZE 128 128 $MODE
done
for BATCH_SIZE in ${_I128O2048_BATCH_SIZE_LIST[@]}; do
    unittest 7 1 $BATCH_SIZE 128 2048 $MODE
done
for BATCH_SIZE in ${_I2048O128_BATCH_SIZE_LIST[@]}; do
    unittest 7 1 $BATCH_SIZE 2048 128 $MODE
done
for BATCH_SIZE in ${_I2048O2048_BATCH_SIZE_LIST[@]}; do
    unittest 7 1 $BATCH_SIZE 2048 2048 $MODE
done
for BATCH_SIZE in ${_I16KO128_BATCH_SIZE_LIST[@]}; do
    unittest 7 1 $BATCH_SIZE 16384 128 $MODE
done
for BATCH_SIZE in ${_I32KO128_BATCH_SIZE_LIST[@]}; do
    unittest 7 1 $BATCH_SIZE 32768 128 $MODE
done
for BATCH_SIZE in ${_I64KO128_BATCH_SIZE_LIST[@]}; do
    unittest 7 1 $BATCH_SIZE 65536 128 $MODE
done
for BATCH_SIZE in ${_I128KO128_BATCH_SIZE_LIST[@]}; do
    unittest 7 1 $BATCH_SIZE 131072 128 $MODE
done


for BATCH_SIZE in ${_I8O256_BATCH_SIZE_LIST[@]}; do
    unittest 13 1 $BATCH_SIZE 8 256 $MODE
done
for BATCH_SIZE in ${_I256O512_BATCH_SIZE_LIST[@]}; do
    unittest 13 1 $BATCH_SIZE 256 512 $MODE
done
for BATCH_SIZE in ${_I512O512_BATCH_SIZE_LIST[@]}; do
    unittest 13 1 $BATCH_SIZE 512 512 $MODE
done
for BATCH_SIZE in ${_I1024O512_BATCH_SIZE_LIST[@]}; do
    unittest 13 1 $BATCH_SIZE 1024 512 $MODE
done
for BATCH_SIZE in ${_I1024O1024_BATCH_SIZE_LIST[@]}; do
    unittest 13 1 $BATCH_SIZE 1024 1024 $MODE
done
for BATCH_SIZE in ${_I128O128_BATCH_SIZE_LIST[@]}; do
    unittest 13 1 $BATCH_SIZE 128 128 $MODE
done
for BATCH_SIZE in ${_I128O2048_BATCH_SIZE_LIST[@]}; do
    unittest 13 1 $BATCH_SIZE 128 2048 $MODE
done
for BATCH_SIZE in ${_I2048O128_BATCH_SIZE_LIST[@]}; do
    unittest 13 1 $BATCH_SIZE 2048 128 $MODE
done
for BATCH_SIZE in ${_I2048O2048_BATCH_SIZE_LIST[@]}; do
    unittest 13 1 $BATCH_SIZE 2048 2048 $MODE
done
for BATCH_SIZE in ${_I16KO128_BATCH_SIZE_LIST[@]}; do
    unittest 13 1 $BATCH_SIZE 16384 128 $MODE
done
for BATCH_SIZE in ${_I32KO128_BATCH_SIZE_LIST[@]}; do
    unittest 13 1 $BATCH_SIZE 32768 128 $MODE
done
for BATCH_SIZE in ${_I64KO128_BATCH_SIZE_LIST[@]}; do
    unittest 13 1 $BATCH_SIZE 65536 128 $MODE
done
for BATCH_SIZE in ${_I128KO128_BATCH_SIZE_LIST[@]}; do
    unittest 13 1 $BATCH_SIZE 131072 128 $MODE
done


for BATCH_SIZE in ${_I8O256_BATCH_SIZE_LIST[@]}; do
    unittest 13 2 $BATCH_SIZE 8 256 $MODE
done
for BATCH_SIZE in ${_I256O512_BATCH_SIZE_LIST[@]}; do
    unittest 13 2 $BATCH_SIZE 256 512 $MODE
done
for BATCH_SIZE in ${_I512O512_BATCH_SIZE_LIST[@]}; do
    unittest 13 2 $BATCH_SIZE 512 512 $MODE
done
for BATCH_SIZE in ${_I1024O512_BATCH_SIZE_LIST[@]}; do
    unittest 13 2 $BATCH_SIZE 1024 512 $MODE
done
for BATCH_SIZE in ${_I1024O1024_BATCH_SIZE_LIST[@]}; do
    unittest 13 2 $BATCH_SIZE 1024 1024 $MODE
done
for BATCH_SIZE in ${_I128O128_BATCH_SIZE_LIST[@]}; do
    unittest 13 2 $BATCH_SIZE 128 128 $MODE
done
for BATCH_SIZE in ${_I128O2048_BATCH_SIZE_LIST[@]}; do
    unittest 13 2 $BATCH_SIZE 128 2048 $MODE
done
for BATCH_SIZE in ${_I2048O128_BATCH_SIZE_LIST[@]}; do
    unittest 13 2 $BATCH_SIZE 2048 128 $MODE
done
for BATCH_SIZE in ${_I2048O2048_BATCH_SIZE_LIST[@]}; do
    unittest 13 2 $BATCH_SIZE 2048 2048 $MODE
done
for BATCH_SIZE in ${_I16KO128_BATCH_SIZE_LIST[@]}; do
    unittest 13 2 $BATCH_SIZE 16384 128 $MODE
done
for BATCH_SIZE in ${_I32KO128_BATCH_SIZE_LIST[@]}; do
    unittest 13 2 $BATCH_SIZE 32768 128 $MODE
done
for BATCH_SIZE in ${_I64KO128_BATCH_SIZE_LIST[@]}; do
    unittest 13 2 $BATCH_SIZE 65536 128 $MODE
done
for BATCH_SIZE in ${_I128KO128_BATCH_SIZE_LIST[@]}; do
    unittest 13 2 $BATCH_SIZE 131072 128 $MODE
done


for BATCH_SIZE in ${_I8O256_BATCH_SIZE_LIST[@]}; do
    unittest 65 8 $BATCH_SIZE 8 256 $MODE
done
for BATCH_SIZE in ${_I256O512_BATCH_SIZE_LIST[@]}; do
    unittest 65 8 $BATCH_SIZE 256 512 $MODE
done
for BATCH_SIZE in ${_I512O512_BATCH_SIZE_LIST[@]}; do
    unittest 65 8 $BATCH_SIZE 512 512 $MODE
done
for BATCH_SIZE in ${_I1024O512_BATCH_SIZE_LIST[@]}; do
    unittest 65 8 $BATCH_SIZE 1024 512 $MODE
done
for BATCH_SIZE in ${_I1024O1024_BATCH_SIZE_LIST[@]}; do
    unittest 65 8 $BATCH_SIZE 1024 1024 $MODE
done
for BATCH_SIZE in ${_I128O128_BATCH_SIZE_LIST[@]}; do
    unittest 65 8 $BATCH_SIZE 128 128 $MODE
done
for BATCH_SIZE in ${_I128O2048_BATCH_SIZE_LIST[@]}; do
    unittest 65 8 $BATCH_SIZE 128 2048 $MODE
done
for BATCH_SIZE in ${_I2048O128_BATCH_SIZE_LIST[@]}; do
    unittest 65 8 $BATCH_SIZE 2048 128 $MODE
done
for BATCH_SIZE in ${_I2048O2048_BATCH_SIZE_LIST[@]}; do
    unittest 65 8 $BATCH_SIZE 2048 2048 $MODE
done
for BATCH_SIZE in ${_I16KO128_BATCH_SIZE_LIST[@]}; do
    unittest 65 8 $BATCH_SIZE 16384 128 $MODE
done
for BATCH_SIZE in ${_I32KO128_BATCH_SIZE_LIST[@]}; do
    unittest 65 8 $BATCH_SIZE 32768 128 $MODE
done
for BATCH_SIZE in ${_I64KO128_BATCH_SIZE_LIST[@]}; do
    unittest 65 8 $BATCH_SIZE 65536 128 $MODE
done
for BATCH_SIZE in ${_I128KO128_BATCH_SIZE_LIST[@]}; do
    unittest 65 8 $BATCH_SIZE 131072 128 $MODE
done


for BATCH_SIZE in ${_I8O256_BATCH_SIZE_LIST[@]}; do
    unittest 70 4 $BATCH_SIZE 8 256 $MODE
done
for BATCH_SIZE in ${_I256O512_BATCH_SIZE_LIST[@]}; do
    unittest 70 4 $BATCH_SIZE 256 512 $MODE
done
for BATCH_SIZE in ${_I512O512_BATCH_SIZE_LIST[@]}; do
    unittest 70 4 $BATCH_SIZE 512 512 $MODE
done
for BATCH_SIZE in ${_I1024O512_BATCH_SIZE_LIST[@]}; do
    unittest 70 4 $BATCH_SIZE 1024 512 $MODE
done
for BATCH_SIZE in ${_I1024O1024_BATCH_SIZE_LIST[@]}; do
    unittest 70 4 $BATCH_SIZE 1024 1024 $MODE
done
for BATCH_SIZE in ${_I128O128_BATCH_SIZE_LIST[@]}; do
    unittest 70 4 $BATCH_SIZE 128 128 $MODE
done
for BATCH_SIZE in ${_I128O2048_BATCH_SIZE_LIST[@]}; do
    unittest 70 4 $BATCH_SIZE 128 2048 $MODE
done
for BATCH_SIZE in ${_I2048O128_BATCH_SIZE_LIST[@]}; do
    unittest 70 4 $BATCH_SIZE 2048 128 $MODE
done
for BATCH_SIZE in ${_I2048O2048_BATCH_SIZE_LIST[@]}; do
    unittest 70 4 $BATCH_SIZE 2048 2048 $MODE
done
for BATCH_SIZE in ${_I16KO128_BATCH_SIZE_LIST[@]}; do
    unittest 70 4 $BATCH_SIZE 16384 128 $MODE
done
for BATCH_SIZE in ${_I32KO128_BATCH_SIZE_LIST[@]}; do
    unittest 70 4 $BATCH_SIZE 32768 128 $MODE
done
for BATCH_SIZE in ${_I64KO128_BATCH_SIZE_LIST[@]}; do
    unittest 70 4 $BATCH_SIZE 65536 128 $MODE
done
for BATCH_SIZE in ${_I128KO128_BATCH_SIZE_LIST[@]}; do
    unittest 70 4 $BATCH_SIZE 131072 128 $MODE
done


for BATCH_SIZE in ${_I8O256_BATCH_SIZE_LIST[@]}; do
    unittest 70 8 $BATCH_SIZE 8 256 $MODE
done
for BATCH_SIZE in ${_I256O512_BATCH_SIZE_LIST[@]}; do
    unittest 70 8 $BATCH_SIZE 256 512 $MODE
done
for BATCH_SIZE in ${_I512O512_BATCH_SIZE_LIST[@]}; do
    unittest 70 8 $BATCH_SIZE 512 512 $MODE
done
for BATCH_SIZE in ${_I1024O512_BATCH_SIZE_LIST[@]}; do
    unittest 70 8 $BATCH_SIZE 1024 512 $MODE
done
for BATCH_SIZE in ${_I1024O1024_BATCH_SIZE_LIST[@]}; do
    unittest 70 8 $BATCH_SIZE 1024 1024 $MODE
done
for BATCH_SIZE in ${_I128O128_BATCH_SIZE_LIST[@]}; do
    unittest 70 8 $BATCH_SIZE 128 128 $MODE
done
for BATCH_SIZE in ${_I128O2048_BATCH_SIZE_LIST[@]}; do
    unittest 70 8 $BATCH_SIZE 128 2048 $MODE
done
for BATCH_SIZE in ${_I2048O128_BATCH_SIZE_LIST[@]}; do
    unittest 70 8 $BATCH_SIZE 2048 128 $MODE
done
for BATCH_SIZE in ${_I2048O2048_BATCH_SIZE_LIST[@]}; do
    unittest 70 8 $BATCH_SIZE 2048 2048 $MODE
done
for BATCH_SIZE in ${_I16KO128_BATCH_SIZE_LIST[@]}; do
    unittest 70 8 $BATCH_SIZE 16384 128 $MODE
done
for BATCH_SIZE in ${_I32KO128_BATCH_SIZE_LIST[@]}; do
    unittest 70 8 $BATCH_SIZE 32768 128 $MODE
done
for BATCH_SIZE in ${_I64KO128_BATCH_SIZE_LIST[@]}; do
    unittest 70 8 $BATCH_SIZE 65536 128 $MODE
done
for BATCH_SIZE in ${_I128KO128_BATCH_SIZE_LIST[@]}; do
    unittest 70 8 $BATCH_SIZE 131072 128 $MODE
done


for BATCH_SIZE in ${_I8O256_BATCH_SIZE_LIST[@]}; do
    unittest 100 4 $BATCH_SIZE 8 256 $MODE
done
for BATCH_SIZE in ${_I256O512_BATCH_SIZE_LIST[@]}; do
    unittest 100 4 $BATCH_SIZE 256 512 $MODE
done
for BATCH_SIZE in ${_I512O512_BATCH_SIZE_LIST[@]}; do
    unittest 100 4 $BATCH_SIZE 512 512 $MODE
done
for BATCH_SIZE in ${_I1024O512_BATCH_SIZE_LIST[@]}; do
    unittest 100 4 $BATCH_SIZE 1024 512 $MODE
done
for BATCH_SIZE in ${_I1024O1024_BATCH_SIZE_LIST[@]}; do
    unittest 100 4 $BATCH_SIZE 1024 1024 $MODE
done
for BATCH_SIZE in ${_I128O128_BATCH_SIZE_LIST[@]}; do
    unittest 100 4 $BATCH_SIZE 128 128 $MODE
done
for BATCH_SIZE in ${_I128O2048_BATCH_SIZE_LIST[@]}; do
    unittest 100 4 $BATCH_SIZE 128 2048 $MODE
done
for BATCH_SIZE in ${_I2048O128_BATCH_SIZE_LIST[@]}; do
    unittest 100 4 $BATCH_SIZE 2048 128 $MODE
done
for BATCH_SIZE in ${_I2048O2048_BATCH_SIZE_LIST[@]}; do
    unittest 100 4 $BATCH_SIZE 2048 2048 $MODE
done
for BATCH_SIZE in ${_I16KO128_BATCH_SIZE_LIST[@]}; do
    unittest 100 4 $BATCH_SIZE 16384 128 $MODE
done
for BATCH_SIZE in ${_I32KO128_BATCH_SIZE_LIST[@]}; do
    unittest 100 4 $BATCH_SIZE 32768 128 $MODE
done
for BATCH_SIZE in ${_I64KO128_BATCH_SIZE_LIST[@]}; do
    unittest 100 4 $BATCH_SIZE 65536 128 $MODE
done
for BATCH_SIZE in ${_I128KO128_BATCH_SIZE_LIST[@]}; do
    unittest 100 4 $BATCH_SIZE 131072 128 $MODE
done


for BATCH_SIZE in ${_I8O256_BATCH_SIZE_LIST[@]}; do
    unittest 100 8 $BATCH_SIZE 8 256 $MODE
done
for BATCH_SIZE in ${_I256O512_BATCH_SIZE_LIST[@]}; do
    unittest 100 8 $BATCH_SIZE 256 512 $MODE
done
for BATCH_SIZE in ${_I512O512_BATCH_SIZE_LIST[@]}; do
    unittest 100 8 $BATCH_SIZE 512 512 $MODE
done
for BATCH_SIZE in ${_I1024O512_BATCH_SIZE_LIST[@]}; do
    unittest 100 8 $BATCH_SIZE 1024 512 $MODE
done
for BATCH_SIZE in ${_I1024O1024_BATCH_SIZE_LIST[@]}; do
    unittest 100 8 $BATCH_SIZE 1024 1024 $MODE
done
for BATCH_SIZE in ${_I128O128_BATCH_SIZE_LIST[@]}; do
    unittest 100 8 $BATCH_SIZE 128 128 $MODE
done
for BATCH_SIZE in ${_I128O2048_BATCH_SIZE_LIST[@]}; do
    unittest 100 8 $BATCH_SIZE 128 2048 $MODE
done
for BATCH_SIZE in ${_I2048O128_BATCH_SIZE_LIST[@]}; do
    unittest 100 8 $BATCH_SIZE 2048 128 $MODE
done
for BATCH_SIZE in ${_I2048O2048_BATCH_SIZE_LIST[@]}; do
    unittest 100 8 $BATCH_SIZE 2048 2048 $MODE
done
for BATCH_SIZE in ${_I16KO128_BATCH_SIZE_LIST[@]}; do
    unittest 100 8 $BATCH_SIZE 16384 128 $MODE
done
for BATCH_SIZE in ${_I32KO128_BATCH_SIZE_LIST[@]}; do
    unittest 100 8 $BATCH_SIZE 32768 128 $MODE
done
for BATCH_SIZE in ${_I64KO128_BATCH_SIZE_LIST[@]}; do
    unittest 100 8 $BATCH_SIZE 65536 128 $MODE
done
for BATCH_SIZE in ${_I128KO128_BATCH_SIZE_LIST[@]}; do
    unittest 100 8 $BATCH_SIZE 131072 128 $MODE
done


done
