#!/bin/bash

BUILD_DIR="../build"
EXECUTABLE="npu-test_matmul_e4m3q"
scp -P 9922 "${BUILD_DIR}/${EXECUTABLE}"  iprc@127.0.0.1:~/test_dir
ssh -p 9922 iprc@127.0.0.1 "cd ~/test_dir && chmod +x ${EXECUTABLE} && ./${EXECUTABLE}"
