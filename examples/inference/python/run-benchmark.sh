#!/bin/bash
# generate input files
weightpath=/data00/home/wangxingyao/projects/lightseq/examples/inference/python/q2q_transformer.pb
testpath=/tmp/lightseq/q2qtest
executable=/tmp/lightseq/q2qtest/transformer_example
echo "testing using weightpath " $weightpath
echo "testing using executable " $executable

run_benchmark() {
    # function arguments: batch_size, batch_seq_len
    python3 generate-cpp-input-file.py $testpath/input-$1-$2.txt $1 $2 >/dev/null
    echo -en "${1}\t${2}\t"
    # executable arguments: model_weights_path, input_file_name
    ($executable $weightpath $testpath/input-$1-$2.txt | grep time | awk -F ":" '{print $2}')
}

# run executatble
echo "==== benchmark result (in ms) ===="
echo "batch_size, batch_seq_len, time(ms)"
run_benchmark 1 32
run_benchmark 1 64
run_benchmark 8 32
run_benchmark 8 64
run_benchmark 32 32
run_benchmark 32 64
run_benchmark 64 32
run_benchmark 64 64
run_benchmark 128 32
run_benchmark 128 64
