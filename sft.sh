export DISABLE_VERSION_CHECK=1
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,3,4,7
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
# ---------------------- Industrial_and_Scientific ----------------------
# --- 0.5B ---
# llamafactory-cli train examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_0.5b_dsz0.yaml
# llamafactory-cli train examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_0.5b_dsz2.yaml

# 自己的数据集第一次
llamafactory-cli train examples/train_full/Industrial_and_Scientific/new_industry_rec_full_sft_0.5b_dsz2.yaml

# llamafactory-cli train examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_0.5b_dsz3.yaml

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
