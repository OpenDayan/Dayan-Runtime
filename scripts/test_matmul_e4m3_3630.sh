#!/bin/bash


BUILD_DIR="../build"
EXECUTABLE="npu-test_matmul_e4m3"
REMOTE_HOST="3630"
REMOTE_DIR="~/test_dir"

# copy to remote
rsync "${BUILD_DIR}/${EXECUTABLE}" "${REMOTE_HOST}:${REMOTE_DIR}/"
if [ $? -ne 0 ]; then
    echo "rsync failed. Please check the connection and paths."
    exit 1
fi


ssh "${REMOTE_HOST}" -t "cd ${REMOTE_DIR} && chmod +x ${EXECUTABLE} && ./${EXECUTABLE}"

