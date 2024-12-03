#!/bin/bash

set -euxo pipefail
#

# decoding from ${INPUT_DIR}/*.txt to ${OUTPUT_DIR}/*.{txt,ann}

# 1. read from input
mkdir -p ${OUTPUT_DIR}
python3 -m mspx.tools.al.utils_brat cmd:b2z input_path:${INPUT_DIR}/ output_path:${OUTPUT_DIR}/input.json delete_nils:1 convert.toker:nltk

# 2. decode
python3 -m mspx.tasks.zrel.main ${MODEL_DIR}/_conf model_load_name:${MODEL_DIR}/zmodel.best.m vocab_load_dir:${VOCAB_DIR}/ log_stderr:1 fs:test d_input_dir: test1.group_files: test0.test_streaming:5 test0.group_files:${OUTPUT_DIR}/input.json test0.output_file:${OUTPUT_DIR}/output.json $EXTRA_ARGS

# 3. convert to brat format
python3 -m mspx.tools.al.utils_brat cmd:z2b input_path:${OUTPUT_DIR}/output.json output_path:${OUTPUT_DIR}/ delete_nils:1
