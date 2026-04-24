#!/bin/bash
set -euo pipefail

# ===================== 配置区域 =====================

# 【全局执行参数: START_STEP】
# 决定从哪个步骤开始运行:
#   1 = 数据处理 (possess_data.py)
#   2 = 抽取多模态特征 / Text2Emb (multimodal_amazon_text2emb.py)
#   3 = 训练 RQ-VAE 模型 (rqvae.py)
#   4 = 生成向量索引 (generate_indices.py)
#   5 = 预处理 SFT 训练数据 (preprocess_data_sft_rl.py)
#   6 = 启动 SFT 训练于评估流 (bash sft.sh & evaluate.sh)
START_STEP=2

export CATEGORY="Industrial_and_Scientific"
POSSESS_DATA_OUTPATH="./data/Amazon" 
export CUDA_VISIBLE="1,3,5,6"
MONO_CUDA="1" 

# 【SFT 与 Evaluate 核心参数】
export Optimization="dsz3"
export SFT_YAML="examples/train_full/${CATEGORY}/new_industry_rec_full_sft_0.5b_${Optimization}.yaml"
export EVAL_EXP_NAME="newsaves/qwen2.5-0.5b/full/${CATEGORY}-sft-${Optimization}"
# ================= ⚡ 核心模态控制器 =================
# 原有可选值: multimodal, text_only, image_only, concat, image2text, image2text_supp, moe_decouple, contrastive_adapter
# 🔥新增集成协同信息可选值: multimodal_collab, concat_collab, moe_collab
export MODAL_CHOICE="moe_collab"
# 补充调节参数
export TEXT_WEIGHT="-1.0"          # -1.0为不用人工加权比重调控，使用原生融合
export COMBINE_ORIG_TEXT="true"    # 如果是 image2text，是否加和原文本
export MOE_EPOCHS=10000             # 当模式为 moe/contrastive 时，专网的训练轮数 (建议对比学习30~50轮即可)

# 🔥 协同提取参数配置 (调参提取到此处)
export SASREC_EPOCHS=10000
export SASREC_BATCH_SIZE=128

SAMPLE_SIZE=5000
USER_K=2
ITEM_K=2

export EVAL_CUDA_LIST="${CUDA_VISIBLE//,/ }" 
export OUTPUT_DIR="./data/${CATEGORY}"
export TASK4_SAMPLE=-1 
export SEED=42

TEXT2EMB_MODEL="/root/sunyuqi/models/Qwen3-VL-8B-Instruct"
MODEL_NAME="Qwen3VL"

# ===================== 函数定义 =====================
red() { echo -e "\033[31m$@\033[0m"; }
green() { echo -e "\033[32m$@\033[0m"; }
yellow() { echo -e "\033[33m$@\033[0m"; }
blue() { echo -e "\033[34m$@\033[0m"; }

check_command() {
    if ! command -v $1 &> /dev/null; then
        red "错误: 命令 '$1' 未找到"
        exit 1
    fi
}

check_path() {
    if [ ! -e "$1" ]; then
        red "错误: 路径不存在: $1"
        exit 1
    fi
}

# ================= 时长统计相关工具 =================
declare -a STEP_NAMES=()
declare -a STEP_TIMES=()

record_step_time() {
    local step_name=$1
    local step_start=$2
    local step_end=$3
    local duration=$((step_end - step_start))
    # 动态追加到数组里
    STEP_NAMES+=("$step_name")
    STEP_TIMES+=("$duration")
}

print_summary_table() {
    local total_script_time=$1
    echo ""
    blue "============================================================"
    blue "                     🚀 全流程耗时统计看板                  "
    blue "============================================================"
    printf "\033[36m%-32s | %-20s\033[0m\n" "执行阶段名称" "耗时 (时:分:秒)"
    blue "------------------------------------------------------------"
    
    for i in "${!STEP_NAMES[@]}"; do
        local name="${STEP_NAMES[$i]}"
        local t="${STEP_TIMES[$i]}"
        local h=$((t / 3600))
        local m=$(( (t % 3600) / 60 ))
        local s=$((t % 60))
        printf "%-30s | %02d:%02d:%02d\n" "$name" "$h" "$m" "$s"
    done
    
    blue "============================================================"
    local th=$((total_script_time / 3600))
    local tm=$(( (total_script_time % 3600) / 60 ))
    local ts=$((total_script_time % 60))
    printf "\033[32m%-30s | %02d:%02d:%02d\033[0m\n" "✅ 脚本总时长" "$th" "$tm" "$ts"
    blue "============================================================"
    echo ""
}

# ===================== 主程序 =====================
main()
{
    blue "========================================"
    blue "开始处理类别: $CATEGORY"
    yellow "=> 当前使用模态模式: [ $MODAL_CHOICE ]"
    yellow "=> 当前设置：从第 [ $START_STEP ] 步开始运行"
    blue "========================================"

    ROOT="${POSSESS_DATA_OUTPATH}/${CATEGORY}"
    NPY_FILE="${ROOT}/${CATEGORY}.emb-${MODEL_NAME}-${MODAL_CHOICE}.npy"

    # ================= [步骤 1] 数据处理 =================
    if [ "$START_STEP" -le 1 ]; then
        t_start=$(date +%s)
        green "[阶段1] 数据处理..." 
        python ./data/possess_data.py \
            --category ${CATEGORY} \
            --sample_size ${SAMPLE_SIZE} \
            --user_k ${USER_K} \
            --item_k ${ITEM_K} \
            --output_path ${POSSESS_DATA_OUTPATH}
            
        if [ ! -d "$ROOT" ]; then
            red "错误: 数据目录未创建: $ROOT"
            exit 1
        fi
        green "✓ 数据处理完成"
        record_step_time "阶段1: 数据处理" $t_start $(date +%s)
    else
        yellow "跳过 [阶段1] 数据处理"
    fi
    # ================= [步骤 1.5] SASRec 提取协同特征 =================
    if [ "$START_STEP" -le 1 ] || [ "$START_STEP" == "1.5" ]; then
        if [[ "$MODAL_CHOICE" == *"collab"* ]]; then
            t_start=$(date +%s)
            green "[阶段1.5] 执行 SASRec 协同信息提取..."
            SASREC_SAVE_DIR="${POSSESS_DATA_OUTPATH}/${CATEGORY}/sasrec_models"
            check_command python
            
            # 执行 Runner 任务，并通过 --save_dir 强制使特征导出到明确指定的目录下
            bash scripts/rec_zoo/train_sasrec.sh \
                --data_dir ${POSSESS_DATA_OUTPATH}/${CATEGORY} \
                --epoch ${SASREC_EPOCHS} \
                --batch_size ${SASREC_BATCH_SIZE} \
                --save_dir ${SASREC_SAVE_DIR}
                
            export COLLAB_EMB_PATH="${SASREC_SAVE_DIR}/sasrec_item_emb.npy"
            if [ ! -f "$COLLAB_EMB_PATH" ]; then
                red "错误: SASRec协同Embedding文件未生成，可能是由于提前退出: $COLLAB_EMB_PATH"
                exit 1
            fi
            green "✓ SASRec协同特征已生成"
            record_step_time "阶段1.5: 提取协同特征(SASRec)" $t_start $(date +%s)
        else
            yellow "跳过 [阶段1.5] 协同特征提取 (当前模式 $MODAL_CHOICE 不需要协同信息)"
        fi
    fi
    # ================= [步骤 2] text2emb =================
    if [ "$START_STEP" -le 2 ]; then
        t_start=$(date +%s)
        green "[阶段2] 提取特征 (当前模式: $MODAL_CHOICE)..."
        check_path "$TEXT2EMB_MODEL"

        # 🌟 1. 无缝搬运 SFT 中能跑通的 NCCL 防卡死环境变量
        export NCCL_DEBUG=INFO
        # export NCCL_SOCKET_IFNAME=eth0       # 强制指定在 eth0 建立多卡通讯端口，防止网络迷路
        # export NCCL_IB_DISABLE=1             # 禁用 InfiniBand 强制走普通 Socket
        export TORCH_NCCL_BLOCKING_WAIT=1    # [关键] 设置阻塞监听，一旦失败立刻报错而不是一直傻等
        export NCCL_CUMEM_HOST_ENABLE=0
        export NCCL_SHM_DISABLE=1            # [最核心] 禁用共享内存！解决 Docker 环境下 /dev/shm 太小造成的沉默卡死
        export LC_ALL=en_US.UTF-8 
        export LC_CTYPE=en_US.UTF-8
        export NCCL_P2P_DISABLE=1   
         # 给端口加个随机数，防范僵尸进程霸占端口
        RANDOM_PORT=$(( 29000 + RANDOM % 1000 ))
        # 确保显卡可见配置生效
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE}

        # 🚀 2. 构建安全的 Bash 数组参数，杜绝参数漏传
        CMD_ARGS=(
            "--dataset" "${CATEGORY}"
            "--root" "${ROOT}"
            "--plm_checkpoint" "${TEXT2EMB_MODEL}"
            "--batch_size" "1"
            "--max_sent_len" "1024"
            "--mode" "${MODAL_CHOICE}"
            "--moe_epochs" "${MOE_EPOCHS}"
            "--text_weight" "${TEXT_WEIGHT}"
        )

        if [ "$COMBINE_ORIG_TEXT" = "true" ]; then
            CMD_ARGS+=("--combine_orig_text")
        fi

        # 🎯 3. 自动绑定协同特征路径
        if [[ "$MODAL_CHOICE" == *"collab"* ]]; then
            SASREC_SAVE_DIR="${POSSESS_DATA_OUTPATH}/${CATEGORY}/sasrec_models"
            COLLAB_EMB_PATH="${SASREC_SAVE_DIR}/sasrec_item_emb.npy"
            CMD_ARGS+=("--collab_emb_path" "${COLLAB_EMB_PATH}")
            yellow "-> 已动态注入参数: --collab_emb_path ${COLLAB_EMB_PATH}"
        fi

        # 🌟 4. 获取显卡数量并使用 accelerate 发起多卡运行
        NUM_GPUS=$(echo $CUDA_VISIBLE | awk -F',' '{print NF}')
        if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -lt 1 ]; then
            NUM_GPUS=1
        fi

        yellow "-> 检测到 ${NUM_GPUS} 张指定显卡，正在套用 SFT 防卡死配置启动多进程加速..."
        
        # 启动 accelerate，注入 SFT 环境
        # 使用最底层的 torchrun 绕开一切包装器带来的探针扫描死锁
        torchrun \
            --nnodes=1 \
            --nproc_per_node=${NUM_GPUS} \
            --master_port=${RANDOM_PORT} \
            ./rq/text2emb/multimodal_amazon_text2emb.py "${CMD_ARGS[@]}"
        # 运行后检查
        if [ ! -f "$NPY_FILE" ]; then
            red "错误: 嵌入文件未生成: $NPY_FILE"
            exit 1
        fi
        green "✓ 嵌入转换完成: $(du -h "$NPY_FILE" | cut -f1)"
        record_step_time "阶段2: 抽多模态/文本特征" $t_start $(date +%s)
    else
        yellow "跳过 [阶段2] text2emb"
    fi

    # ================= [步骤 3] RQ-VAE =================
    if [ "$START_STEP" -le 3 ]; then
        t_start=$(date +%s)
        yellow "清理旧模型与参数..."
        rm -rf "./rq_output/${CATEGORY}"
        mkdir -p "./rq_output/${CATEGORY}"

        green "[阶段3] RQ-VAE 训练..."
        python rq/rqvae.py \
          --data_path ${NPY_FILE} \
          --ckpt_dir ./rq_output/${CATEGORY} \
          --lr 1e-3 \
          --num_emb_list 256 256 256 256 \
          --sk_epsilons 0.0 0.0 0.0 0.0 \
          --epochs 5000 \
          --batch_size 20480
          
        if [ ! -f "./rq_output/${CATEGORY}/best_collision_model.pth" ]; then
            red "错误: RQ-VAE 模型未生成"
            exit 1
        fi
        green "✓ RQ-VAE 训练完成"
        record_step_time "阶段3: RQ-VAE 训练" $t_start $(date +%s)
    else
        yellow "跳过 [阶段3] RQ-VAE 训练"
    fi

    # ================= [步骤 4] 生成索引 =================
    if [ "$START_STEP" -le 4 ]; then
        t_start=$(date +%s)
        rm -rf ./data/Amazon/Industrial_and_Scientific/*.index.json
        green "[阶段4] 生成索引..."
        python rq/generate_indices.py \
            --dataset ${CATEGORY} \
            --ckpt_path ./rq_output/${CATEGORY}/best_collision_model.pth \
            --output_dir ${POSSESS_DATA_OUTPATH}/${CATEGORY} \
            --device cuda:${MONO_CUDA}
        green "✓ 索引生成完成"
        record_step_time "阶段4: 生成 Codebook 索引" $t_start $(date +%s)
    else
        yellow "跳过 [阶段4] 生成索引"
    fi

    # ================= [步骤 5] 预处理 SFT/RL 数据 =================
    if [ "$START_STEP" -le 5 ]; then
        t_start=$(date +%s)
        yellow "清理历史 SFT / RL 数据集中..."
        rm -rf "$OUTPUT_DIR/sft" "$OUTPUT_DIR/rl"
        mkdir -p "$OUTPUT_DIR"
        
        green "[阶段5] 预处理 SFT/RL 数据..."
        
        # 统一变量定义，不再硬编码 image2text
        ITEMJSON_PATH="${POSSESS_DATA_OUTPATH}/${CATEGORY}/${CATEGORY}.item.json"
        ITEM_BAK_PATH="${POSSESS_DATA_OUTPATH}/${CATEGORY}/${CATEGORY}.item.json.bak"
        GENERATED_TXT_PATH="${POSSESS_DATA_OUTPATH}/${CATEGORY}/${CATEGORY}.item-${MODAL_CHOICE}.json"
        
        # 👉 动态热替换机制 (无论是 image2text 还是 image2text_supp 都适用)
        if [[ "$MODAL_CHOICE" == "image2text" || "$MODAL_CHOICE" == "image2text_supp" ]] && [ -f "$GENERATED_TXT_PATH" ]; then
            yellow "🧪 数据调包: 正在将底表切换为 [$MODAL_CHOICE] 专属生成的特征副本..."
            cp "$ITEMJSON_PATH" "$ITEM_BAK_PATH"
            cp "$GENERATED_TXT_PATH" "$ITEMJSON_PATH"
        fi

        python preprocess_data_sft_rl.py \
            --data_dir $POSSESS_DATA_OUTPATH \
            --category $CATEGORY \
            --output_dir $OUTPUT_DIR \
            --seq_sample $TASK4_SAMPLE \
            --seed $SEED \
            --data_source $CATEGORY \
            --sid_num 4
            
        # 👉 无痕恢复机制
        if [[ "$MODAL_CHOICE" == "image2text" || "$MODAL_CHOICE" == "image2text_supp" ]] && [ -f "$ITEM_BAK_PATH" ]; then
            yellow "🧪 安全复原: 正在回滚原始底表，保护原数据环境不受污染!"
            mv "$ITEM_BAK_PATH" "$ITEMJSON_PATH"
        fi
        
        green "✓ 预处理步骤完成!"
        blue "输出目录: $OUTPUT_DIR"
        record_step_time "阶段5: 预处理 SFT/RL 数据" $t_start $(date +%s)
    else
        yellow "跳过 [阶段5] 预处理 SFT/RL 数据"
    fi
    # # ================= [步骤 5.5] 注入视觉结构先验 (修复灾难遗忘问题) =================
    # if [ "$START_STEP" -le 6 ]; then
    #     green "[阶段5.5] 从 RQ-VAE 抽取 Codebook特征 并取代 LLM Embedding 随机初始化..."
        
    #     # 定义我们注入好的基座模型存储位置
    #     export INJECTED_MODEL_DIR="./models/Qwen2.5-0.5B-Inject-${CATEGORY}"
    #     mkdir -p ./models
        
    #     python inject_priors.py \
    #         --model_id "Qwen/Qwen2.5-0.5B-Instruct" \
    #         --new_tokens_path "$OUTPUT_DIR/new_tokens.json" \
    #         --rqvae_ckpt "./rq_output/${CATEGORY}/best_collision_model.pth" \
    #         --output_path ${INJECTED_MODEL_DIR}
            
    #     green "✓ 先验注入完毕。接下来将基于该模型启动 LLaMA-Factory SFT..."
    # fi
    # ================= [步骤 6] 后续 Bash 脚本启动 =================
    if [ "$START_STEP" -le 6 ]; then
        t_start=$(date +%s)
        green "[阶段6] 执行 SFT 与 Evaluate..."
        bash sft.sh
        bash evaluate.sh
        green "✓ 全部流程执行完毕!"
        record_step_time "阶段6: SFT与Eval验证" $t_start $(date +%s)
    else
        yellow "跳过 [阶段6] 后续执行"
    fi
}

start_time=$(date +%s)
check_command python
check_command nvidia-smi 

main

end_time=$(date +%s)
duration=$((end_time - start_time))

# 最后打印精美的时间汇总表格
print_summary_table $duration