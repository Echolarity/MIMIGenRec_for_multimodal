
# ================= 核心参数安全导入与告警机制 =================
get_env_or_default() {
    local env_val="$1"        # 传入的环境变量的值
    local env_name="$2"       # 环境变量的名称 (仅作打印展示)
    local default_val="$3"    # 如果为空时的默认值

    if [ -z "$env_val" ]; then
        # 打印告警到标准错误流 (>&2)，确保不会影响变量捕获，同时在终端显示黄色高亮
        echo -e "\033[33m[Warning] 环境变量 '${env_name}' 为空，使用默认值: ${default_val}\033[0m" >&2
        echo "$default_val"
    else
        echo "$env_val"
    fi
}
# ==============================================================



export DISABLE_VERSION_CHECK=1
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE:-"0,1,3"}
export CUDA_VISIBLE_DEVICES=$(get_env_or_default "${CUDA_VISIBLE:-}" "CUDA_VISIBLE" "0,1,3")
export HF_ENDPOINT=https://hf-mirror.com
# set wandb project
export WANDB_PROJECT=MIMIGenRec-SFT
export WANDB_MODE=offline
# export WANDB_MODE=online
export WANDB_API_KEY=""
set -x
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # 或你的网卡名称
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1

export LC_ALL=en_US.UTF-8 
export LC_CTYPE=en_US.UTF-8
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_SHM_DISABLE=1 
# CATEGORY=${CATEGORY:-"Industrial_and_Scientific"}
export CATEGORY=$(get_env_or_default "${CATEGORY:-}" "CATEGORY" "Industrial_and_Scientific")
# Optimization=${Optimization:-"dsz3"}
export Optimization=$(get_env_or_default "${Optimization:-}" "Optimization" "dsz3")

SFT_YAML_PATH=$(get_env_or_default "${SFT_YAML:-}" "SFT_YAML" "examples/train_full/${CATEGORY}/new_industry_rec_full_sft_0.5b_${Optimization}.yaml")
# SFT_YAML_PATH=${SFT_YAML:-"examples/train_full/${CATEGORY}/new_industry_rec_full_sft_0.5b_${Optimization}.yaml"}
echo "Starting SFT with Config: $SFT_YAML_PATH"
llamafactory-cli train $SFT_YAML_PATH
# ---------------------- Industrial_and_Scientific ----------------------
# --- 0.5B ---
# llamafactory-cli train examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_0.5b_dsz0.yaml
# llamafactory-cli train examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_0.5b_dsz2.yaml

# 自己的数据集第一次
# llamafactory-cli train examples/train_full/Industrial_and_Scientific/new_industry_rec_full_sft_0.5b_dsz2.yaml

# llamafactory-cli train examples/train_full/Industrial_and_Scientific/new_industry_rec_full_sft_0.5b_dsz3.yaml

# --- 1.5B ---
# llamafactory-cli train examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_1.5b_dsz0.yaml
# llamafactory-cli train examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_1.5b_dsz2.yaml
# llamafactory-cli train examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_1.5b_dsz3.yaml

# # --- 3B ---
# llamafactory-cli train examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_3b_dsz0.yaml
# llamafactory-cli train examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_3b_dsz2.yaml
# llamafactory-cli train examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_3b_dsz3.yaml

# ---------------------- Office_Products ----------------------
# --- 0.5B ---
# llamafactory-cli train examples/train_full/Office_Products/office_rec_full_sft_0.5b_dsz0.yaml
# llamafactory-cli train examples/train_full/Office_Products/office_rec_full_sft_0.5b_dsz2.yaml
# llamafactory-cli train examples/train_full/Office_Products/office_rec_full_sft_0.5b_dsz3.yaml

# --- 1.5B ---
# llamafactory-cli train examples/train_full/Office_Products/office_rec_full_sft_1.5b_dsz0.yaml
# llamafactory-cli train examples/train_full/Office_Products/office_rec_full_sft_1.5b_dsz2.yaml
# llamafactory-cli train examples/train_full/Office_Products/office_rec_full_sft_1.5b_dsz3.yaml

# --- 3B ---
# llamafactory-cli train examples/train_full/Office_Products/office_rec_full_sft_3b_dsz0.yaml
# llamafactory-cli train examples/train_full/Office_Products/office_rec_full_sft_3b_dsz2.yaml
# llamafactory-cli train examples/train_full/Office_Products/office_rec_full_sft_3b_dsz3.yaml

# ---------------------- Toys_and_Games ----------------------
# --- 0.5B ---
# llamafactory-cli train examples/train_full/Toys_and_Games/toys_rec_full_sft_0.5b_dsz0.yaml


#第一次跑通
# llamafactory-cli train examples/train_full/Toys_and_Games/toys_rec_full_sft_0.5b_dsz2.yaml

# llamafactory-cli train examples/train_full/Toys_and_Games/toys_rec_full_sft_0.5b_dsz3.yaml

# --- 1.5B ---
# llamafactory-cli train examples/train_full/Toys_and_Games/toys_rec_full_sft_1.5b_dsz0.yaml
# llamafactory-cli train examples/train_full/Toys_and_Games/toys_rec_full_sft_1.5b_dsz2.yaml
# llamafactory-cli train examples/train_full/Toys_and_Games/toys_rec_full_sft_1.5b_dsz3.yaml

# --- 3B ---
# llamafactory-cli train examples/train_full/Toys_and_Games/toys_rec_full_sft_3b_dsz0.yaml
# llamafactory-cli train examples/train_full/Toys_and_Games/toys_rec_full_sft_3b_dsz2.yaml
# llamafactory-cli train examples/train_full/Toys_and_Games/toys_rec_full_sft_3b_dsz3.yaml
