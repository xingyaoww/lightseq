prefixname=$1
set -x
cd /opt/tiger/lightseq/examples/inference/python
python3 -u ls_bart_time_test_with_diff_beamsz.py 4 > $prefixname-4beams-fp16-lstime.log
python3 -u ls_bart_time_test_with_diff_beamsz.py 8 > $prefixname-8beams-fp16-lstime.log
python3 -u ls_bart_time_test_with_diff_beamsz.py 16 > $prefixname-16beams-fp16-lstime.log
python3 -u ls_bart_time_test_with_diff_beamsz.py 32 > $prefixname-32beams-fp16-lstime.log