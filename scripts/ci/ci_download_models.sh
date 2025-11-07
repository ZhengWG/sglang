#!/bin/bash

set -euxo pipefail

DEFAULT_MODEL_CACHE_DIR=${DEFAULT_MODEL_CACHE_DIR:-/home/shared/models/sglang-cicd-models}
SGLANG_REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}")"/../../ && pwd )"

# key is model path in huggingface form, if value is empty, download from huggingface, else download from modelscope
declare -A model_mapping 


model_mapping["meta-llama/Llama-3.1-8B-Instruct"]="LLM-Research/Meta-Llama-3.1-8B-Instruct"
model_mapping["meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"]="LLM-Research/Llama-4-Maverick-17B-128E-Instruct-FP8"
# TODO make soft link instead of download twice
model_mapping["meta-llama/Meta-Llama-3.1-8B-Instruct"]="LLM-Research/Meta-Llama-3.1-8B-Instruct"
model_mapping["openai/gpt-oss-120b"]="openai-mirror/gpt-oss-120b"
model_mapping["meta-llama/Llama-2-7b-hf"]="shakechen/Llama-2-7b-hf"
model_mapping["Qwen/Qwen3-4B"]="Qwen/Qwen3-4B"
model_mapping["RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8"]="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8"
model_mapping["google/gemma-2-2b"]="google/gemma-2-2b"
model_mapping["Qwen/Qwen2-7B-Instruct"]="Qwen/Qwen2-7B-Instruct"
model_mapping["LxzGordon/URM-LLaMa-3.1-8B"]="AI-ModelScope/URM-LLaMa-3.1-8B"
model_mapping["neuralmagic/Qwen2-7B-Instruct-FP8"]="neuralmagic/Qwen2-7B-Instruct-FP8"
model_mapping["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
model_mapping["google/gemma-3-27b-it"]="google/gemma-3-27b-it"
model_mapping["meta-llama/Llama-2-7b-chat-hf"]="NousResearch/Llama-2-7b-chat-hf"
model_mapping["mistralai/Mistral-7B-Instruct-v0.3"]="mistralai/Mistral-7B-Instruct-v0.3"
model_mapping["Qwen/Qwen2.5-7B-Instruct"]="Qwen/Qwen2.5-7B-Instruct"
model_mapping["Qwen/Qwen3-30B-A3B"]="Qwen/Qwen3-30B-A3B"
model_mapping["RedHatAI/Qwen3-30B-A3B-FP8-dynamic"]="RedHatAI/Qwen3-30B-A3B-FP8-dynamic"
model_mapping["deepseek-ai/deepseek-vl2-small"]="deepseek-ai/deepseek-vl2-small"
model_mapping["deepseek-ai/Janus-Pro-7B"]="deepseek-ai/Janus-Pro-7B"
model_mapping["Efficient-Large-Model/NVILA-Lite-2B-hf-0626"]="Efficient-Large-Model/NVILA-Lite-2B-hf-0626"
model_mapping["google/gemma-3-4b-it"]="google/gemma-3-4b-it"
model_mapping["google/gemma-3n-E4B-it"]="google/gemma-3n-E4B-it"
model_mapping["lmms-lab/llava-onevision-qwen2-0.5b-ov"]="lmms-lab/llava-onevision-qwen2-0.5b-ov"
model_mapping["lmms-lab/llava-onevision-qwen2-7b-ov"]="lmms-lab/llava-onevision-qwen2-7b-ov"
model_mapping["mistral-community/pixtral-12b"]="AI-ModelScope/pixtral-12b"
model_mapping["neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8"]="neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8"
model_mapping["neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"]="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
model_mapping["neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a8"]="neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a8"
model_mapping["openai/gpt-oss-20b"]="openai-mirror/gpt-oss-20b"
model_mapping["openbmb/MiniCPM-V-2_6"]="openbmb/MiniCPM-V-2_6"
model_mapping["Qwen/Qwen1.5-MoE-A2.7B"]="Qwen/Qwen1.5-MoE-A2.7B"
model_mapping["Qwen/Qwen2.5-VL-7B-Instruct"]="Qwen/Qwen2.5-VL-7B-Instruct"
model_mapping["Qwen/Qwen2-Audio-7B-Instruct"]="Qwen/Qwen2-Audio-7B-Instruct"
model_mapping["Qwen/Qwen2-VL-7B-Instruct"]="Qwen/Qwen2-VL-7B-Instruct"
model_mapping["Qwen/Qwen3-0.6B"]="Qwen/Qwen3-0.6B"
model_mapping["Qwen/Qwen3-8B"]="Qwen/Qwen3-8B"
model_mapping["Qwen/Qwen3-Next-80B-A3B-Instruct"]="Qwen/Qwen3-Next-80B-A3B-Instruct"
model_mapping["unsloth/Mistral-Small-3.1-24B-Instruct-2503"]="unsloth/Mistral-Small-3.1-24B-Instruct-2503"
model_mapping["XiaomiMiMo/MiMo-VL-7B-RL"]="XiaomiMiMo/MiMo-VL-7B-RL"
model_mapping["zai-org/GLM-4.1V-9B-Thinking"]="ZhipuAI/GLM-4.1V-9B-Thinking"
model_mapping["RedHatAI/Meta-Llama-3.1-8B-FP8"]="RedHatAI/Meta-Llama-3.1-8B-FP8"

model_mapping["deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"]="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
model_mapping["algoprog/fact-generation-llama-3.1-8b-instruct-lora"]=""
model_mapping["Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16"]=""
model_mapping["winddude/wizardLM-LlaMA-LoRA-7B"]=""
model_mapping["RuterNorway/Llama-2-7b-chat-norwegian-LoRa"]=""
model_mapping["faridlazuarda/valadapt-llama-3.1-8B-it-chinese"]=""
model_mapping["philschmid/code-llama-3-1-8b-text-to-sql-lora"]=""
model_mapping["pbevan11/llama-3.1-8b-ocr-correction"]=""
model_mapping["nissenj/Qwen3-4B-lora-v2"]=""
model_mapping["Alibaba-NLP/gte-Qwen2-1.5B-instruct"]=""
model_mapping["meta-llama/Llama-3.2-1B-Instruct"]="LLM-Research/Llama-3.2-1B-Instruct"
model_mapping["marco/mcdse-2b-v1"]="AI-ModelScope/mcdse-2b-v1"
model_mapping["meta-llama/Llama-3.2-1B"]="LLM-Research/Llama-3.2-1B"
model_mapping["Qwen/Qwen2-1.5B"]="Qwen/Qwen2-1.5B"
model_mapping["Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"]="AI-ModelScope/Skywork-Reward-Llama-3.1-8B-v0.2"
model_mapping["moonshotai/Kimi-VL-A3B-Instruct"]="moonshotai/Kimi-VL-A3B-Instruct"
# TODO add mirror for those two model
#model_mapping["lmsys/sglang-ci-dsv3-test"]="TODO"
#model_mapping["sgl-project/sglang-ci-dsv3-block-int8-test"]="TODO"
#model_mapping["sgl-project/sglang-ci-dsv3-channel-int8-test"]="TODO"
model_mapping["lmsys/gpt-oss-20b-bf16"]=""
model_mapping["lmsys/gpt-oss-120b-bf16"]=""

# download from ais
#model_mapping["lmsys/gpt-oss-20b-bf16"]="ais_id10700008_ver80100042"
model_mapping["BAAI/bge-small-en"]="BAAI/bge-small-en"
model_mapping["cross-encoder/ms-marco-MiniLM-L6-v2"]="cross-encoder/ms-marco-MiniLM-L6-v2"
model_mapping["BAAI/bge-reranker-v2-m3"]="BAAI/bge-reranker-v2-m3"
model_mapping["meta-llama/Llama-4-Scout-17B-16E-Instruct"]="LLM-Research/Llama-4-Scout-17B-16E-Instruct"
model_mapping["meta-llama/Meta-Llama-3-8B-Instruct"]="LLM-Research/Meta-Llama-3-8B-Instruct"
model_mapping["neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"]="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"
model_mapping["openbmb/MiniCPM-o-2_6"]="openbmb/MiniCPM-o-2_6"
model_mapping["openbmb/MiniCPM-V-4"]="openbmb/MiniCPM-V-4"
model_mapping["OpenGVLab/InternVL2_5-2B"]="OpenGVLab/InternVL2_5-2B"
model_mapping["intfloat/e5-mistral-7b-instruct"]="intfloat/e5-mistral-7b-instruct"
model_mapping["y9760210/Qwen3-4B-lora_model"]="ais_id17500029_ver87800081"
model_mapping["jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"]="ais_id17500034_ver87800087"
model_mapping["lmsys/sglang-EAGLE-llama2-chat-7B"]="ais_id17500035_ver87800089"
model_mapping["Barrrrry/DeepSeek-R1-W4AFP8"]="ais_id7600018_ver76700041"
#model_mapping["lmsys/gpt-oss-120b-bf16"]="ais_id10700007_ver80100041"
# 这个可能不对, ais上是 lmsys/DeepSeek-V3-0324-NextN
model_mapping["lmsys/sglang-ci-dsv3-test-NextN"]="ais_id17500036_ver87800090"

function download_model_from_aistudio(){
  # mirror_model should be in format ais_id\d+_ver\d+
  original_model=$1
  mirror_model=$2
  model_id=${mirror_model#*id}
  model_id=${model_id%%_*}
  model_version=${mirror_model#*_ver}
  amax_maas_cli modelhub model_download_with_version_id --amax_modelhub_id ${model_id} -v ${model_version} -s 241025 --enable_mount=False -w -l ${DEFAULT_MODEL_CACHE_DIR}/${original_model}
}

function prepare_model(){
  local original_model=$1
  local mirror_model=${2:-}
  if [ x"${mirror_model}" == x ]; then
    HF_ENDPOINT=https://hf-mirror.com hf download --local-dir ${DEFAULT_MODEL_CACHE_DIR}/${original_model} ${original_model}
  else
    if [[ "${mirror_model}" == ais_** ]]; then
      # 从aistudio下载
      download_model_from_aistudio ${original_model} ${mirror_model}
    else
      # 从modelscope下载
      modelscope download --model ${mirror_model} --local_dir ${DEFAULT_MODEL_CACHE_DIR}/${original_model}
    fi
    if [ $? -ne 0 ];then
      download ${original_model} failed
      rm -vrf ${DEFAULT_MODEL_CACHE_DIR}/${original_model}
    fi
  fi
}

function prepare_models(){
  for model in "${!model_mapping[@]}"; do 
    # 如果已经存在目录，不再重复下载
      prepare_model ${model} ${model_mapping["${model}"]}
  done
}

function prepare_cached_model_links(){
  for model in "${!model_mapping[@]}"; do
    target_dir=${SGLANG_REPO_DIR}/test/srt/$(dirname ${model})
    mkdir -p ${target_dir}
    ln -sf ${DEFAULT_MODEL_CACHE_DIR}/${model} ${target_dir}
  done
}


declare -A dataset_mapping

dataset_mapping["lmms-lab/MMMU"]="HuggingFaceM4/MMMU"

function prepare_datasets(){
  for dataset in "${!dataset_mapping[@]}"; do
    # 如果已经存在目录，不再重复下载
      modelscope download --dataset ${dataset_mapping["${dataset}"]} --local_dir ${DEFAULT_MODEL_CACHE_DIR}/${dataset} \
        --exclude dataset_infos.json
  done
}

function prepare_cached_dataset_links(){
  for dataset in "${!dataset_mapping[@]}"; do
    target_dir=${SGLANG_REPO_DIR}/test/srt/$(dirname ${dataset})
    mkdir -p ${target_dir}
    ln -sf ${DEFAULT_MODEL_CACHE_DIR}/${dataset} ${target_dir}
  done
}


prepare_models 

prepare_cached_model_links

prepare_datasets

prepare_cached_dataset_links

