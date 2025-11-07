#!/bin/bash

SGLANG_REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}")"/../../ && pwd )"
DEFAULT_MODEL_CACHE_DIR=${DEFAULT_MODEL_CACHE_DIR:-/home/local/workspace/models}


function ln_models_to_test_srt(){
  SRC_DIR="$1"
  DEST_DIR="$2"
  find "$SRC_DIR" -mindepth 2 -maxdepth 2 -type d -print0 |
  while IFS= read -r -d '' dir; do
    # 去掉前缀，得到相对路径  A/B
    rel=${dir#"$SRC_DIR"/}
    # 目标目录的父路径
    dest_parent="$DEST_DIR/$(dirname "$rel")"
    # 目标链接完整路径
    dest_link="$DEST_DIR/$rel"
    mkdir -p "$dest_parent"
    # 如果已存在且不是我们的链接，先备份
    if [[ -e "$dest_link" && ! -L "$dest_link" ]]; then
        mv "$dest_link" "${dest_link}.bak.$(date +%s)"
    fi
    # 链接不存在才建
    if [[ ! -e "$dest_link" ]]; then
        # 计算从 dest_link 所在位置到源目录的相对路径
        # 例如 dest/A/B -> ../../source/A/B
        rel_path=$(realpath --relative-to="$dest_parent" "$dir")
        ln -s "$rel_path" "$dest_link"
        echo "linked  $dest_link  ->  $rel_path"
    fi
done
}

ln_models_to_test_srt "${DEFAULT_MODEL_CACHE_DIR}" "${SGLANG_REPO_DIR}/test/srt"

#export DEFAULT_MODEL_CACHE_DIR=${DEFAULT_MODEL_CACHE_DIR}
#export HF_ENDPOINT="https://hf-mirror.com"
#export HF_HUB_OFFLINE=1 
export NCCL_DEBUG=WARN
export SGLANG_IS_IN_CI=1
export SGLANG_TEST_MAX_RETRY=0
#export HF_HOME="${DEFAULT_MODEL_CACHE_DIR}"

cd ${SGLANG_REPO_DIR}/test/srt

mkdir -p ${SGLANG_REPO_DIR}/testresults

TEST_RESULT_DIR="${SGLANG_REPO_DIR}/testresults"

pytest -s --junitxml=${TEST_RESULT_DIR}/results.xml --cov=sglang --cov-branch --cov-config ${SGLANG_REPO_DIR}/.coveragerc --cov-report=xml:${TEST_RESULT_DIR}/coverage.xml run_suite_theta.py

