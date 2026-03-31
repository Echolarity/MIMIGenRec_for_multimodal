#!/bin/bash
set -euo pipefail

# ===================== 配置区域 =====================

# 【全局执行参数: START_STEP】
# 决定从哪个步骤开始运行 (设置数字后，会从该步骤起一直向后运行到结束):
#   1 = 数据处理 (possess_data.py)
#   2 = 抽取多模态特征 / Text2Emb (multimodal_amazon_text2emb.py)
#   3 = 训练 RQ-VAE 模型 (rqvae.py)
#   4 = 生成向量索引 (generate_indices.py)
#   5 = 预处理 SFT 训练数据 (preprocess_data_sft_rl.py)
#   6 = 启动 SFT 训练于评估流 (bash sft.sh & evaluate.sh)
START_STEP=5


export CATEGORY="Industrial_and_Scientific"
POSSESS_DATA_OUTPATH="./data/Amazon" 
export CUDA_VISIBLE="0,1,4,6" # SFT和Emb使用的多卡 (逗号分隔)
MONO_CUDA="3" # 选一个充足的单卡

# 【SFT 与 Evaluate 核心参数统一管理】
export Optimization="dsz3"
export SFT_YAML="examples/train_full/${CATEGORY}/new_industry_rec_full_sft_0.5b_${Optimization}.yaml"
export EVAL_EXP_NAME="newsaves/qwen2.5-0.5b/full/${CATEGORY}-sft-${Optimization}"


SAMPLE_SIZE=5000
USER_K=2
ITEM_K=2
MODAL_CHOICE="multimodal"
# mode写法：choices=["multimodal", "text_only", "image_only", "concat"]
export EVAL_CUDA_LIST="${CUDA_VISIBLE//,/ }"  # 自动把逗号转为空格，供 evaluate.sh 分发使用
# output directory (sft/, rl/, new_tokens.json); default data/<category>
export OUTPUT_DIR="./data/${CATEGORY}"
export TASK4_SAMPLE=-1 # sample all if -1
export SEED=42

TEXT2EMB_MODEL="/root/sunyuqi/models/Qwen3-VL-8B-Instruct"
MODEL_NAME="Qwen3VL"


# ===================== 函数定义 =====================
# 颜色输出函数
red() { echo -e "\033[31m$@\033[0m"; }
green() { echo -e "\033[32m$@\033[0m"; }
yellow() { echo -e "\033[33m$@\033[0m"; }
blue() { echo -e "\033[34m$@\033[0m"; }

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        red "错误: 命令 '$1' 未找到"
        exit 1
    fi
}

# 检查文件/目录是否存在
check_path() {
    if [ ! -e "$1" ]; then
        red "错误: 路径不存在: $1"
        exit 1
    fi
}

# ===================== 主程序 =====================
main()
{
    blue "========================================"
    blue "开始处理类别: $CATEGORY"
    yellow "=> 当前设置：从第 [ $START_STEP ] 步开始向下运行"
    blue "========================================"

    # 【重要说明】将有依赖的路径变量提取到主流程最外层
    # 防止因跳过前置步骤导致后续步骤中这些变量为空
    ROOT="${POSSESS_DATA_OUTPATH}/${CATEGORY}"
    NPY_FILE="${ROOT}/${CATEGORY}.emb-${MODEL_NAME}-${MODAL_CHOICE}.npy"

    # ================= [步骤 1] 数据处理 =================
    if [ "$START_STEP" -le 1 ]; then
        green "[阶段1] 数据处理..." 
        python ./data/possess_data.py \
            --category ${CATEGORY} \
            --sample_size ${SAMPLE_SIZE} \
            --user_k ${USER_K} \
            --item_k ${ITEM_K} \
            --output_path ${POSSESS_DATA_OUTPATH}
            
        # 检查输出目录
        if [ ! -d "$ROOT" ]; then
            red "错误: 数据目录未创建: $ROOT"
            exit 1
        fi
        green "✓ 数据处理完成"
    else
        yellow "跳过 [阶段1] 数据处理"
    fi

    # ================= [步骤 2] text2emb =================
    if [ "$START_STEP" -le 2 ]; then
        green "[阶段2] text2emb..."
        check_path "$TEXT2EMB_MODEL"
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE} python ./rq/text2emb/multimodal_amazon_text2emb.py \
            --dataset ${CATEGORY} \
            --root ${ROOT} \
            --plm_checkpoint ${TEXT2EMB_MODEL} \
            --batch_size 1 \
            --max_sent_len 1024 \
            --mode ${MODAL_CHOICE}
        
        # 检查嵌入文件
        if [ ! -f "$NPY_FILE" ]; then
            red "错误: 嵌入文件未生成: $NPY_FILE"
            exit 1
        fi
        green "✓ 嵌入转换完成: $(du -h "$NPY_FILE" | cut -f1)"
    else
        yellow "跳过 [阶段2] text2emb"
    fi

    # ================= [步骤 3] RQ-VAE =================
    if [ "$START_STEP" -le 3 ]; then
        green "[阶段3] RQ-VAE 训练..."
        mkdir -p "./rq_output/${CATEGORY}"
        python rq/rqvae.py \
          --data_path ${NPY_FILE} \
          --ckpt_dir ./rq_output/${CATEGORY} \
          --lr 1e-3 \
          --epochs 10000 \
          --batch_size 20480
          
        # 检查模型文件
        if [ ! -f "./rq_output/${CATEGORY}/best_collision_model.pth" ]; then
            red "错误: RQ-VAE 模型未生成"
            exit 1
        fi
        green "✓ RQ-VAE 训练完成"
    else
        yellow "跳过 [阶段3] RQ-VAE 训练"
    fi

    # ================= [步骤 4] 生成索引 =================
    if [ "$START_STEP" -le 4 ]; then
        green "[阶段4] 生成索引..."
        python rq/generate_indices.py \
            --dataset ${CATEGORY} \
            --ckpt_path ./rq_output/${CATEGORY}/best_collision_model.pth \
            --output_dir ${POSSESS_DATA_OUTPATH}/${CATEGORY} \
            --device cuda:${MONO_CUDA}
        green "✓ 索引生成完成"
    else
        yellow "跳过 [阶段4] 生成索引"
    fi

    # ================= [步骤 5] 预处理 SFT/RL 数据 =================
    if [ "$START_STEP" -le 5 ]; then
        green "[阶段5] 预处理 SFT/RL 数据..."
        # 创建输出目录
        mkdir -p "$OUTPUT_DIR"
        
        # 导出必要的环境变量
        export OUTPUT_DIR
        export TASK4_SAMPLE
        export SEED
        python preprocess_data_sft_rl.py \
            --data_dir $POSSESS_DATA_OUTPATH \
            --category $CATEGORY \
            --output_dir $OUTPUT_DIR \
            --seq_sample $TASK4_SAMPLE \
            --seed $SEED \
            --data_source $CATEGORY
        
        green "✓ 预处理步骤完成!"
        blue "输出目录: $OUTPUT_DIR"
        
        # 显示最终文件结构
        if [ -d "$OUTPUT_DIR" ]; then
            echo ""
            blue "生成的文件:"
            tree "$OUTPUT_DIR" -L 2 || ls -la "$OUTPUT_DIR/"
        fi
    else
        yellow "跳过 [阶段5] 预处理 SFT/RL 数据"
    fi

    # ================= [步骤 6] 后续 Bash 脚本启动 =================
    if [ "$START_STEP" -le 6 ]; then
        green "[阶段6] 执行 SFT 与 Evaluate..."
        bash sft.sh
        bash evaluate.sh
        green "✓ 全部流程执行完毕!"
    else
        yellow "跳过 [阶段6] 后续执行"
    fi
}
# 记录脚本开始时间
start_time=$(date +%s)
# 环境与依赖检查
check_command python
check_command nvidia-smi  # 检查 GPU 可用性

# 开始执行
main
end_time=$(date +%s)
# 计算运行时长（秒）
duration=$((end_time - start_time))

# 将秒转换为时:分:秒格式
if [ $duration -ge 3600 ]; then
    hours=$((duration / 3600))
    minutes=$(( (duration % 3600) / 60 ))
    seconds=$((duration % 60))
    printf "脚本运行时长: %02d:%02d:%02d (时:分:秒)\n" $hours $minutes $seconds
elif [ $duration -ge 60 ]; then
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    printf "脚本运行时长: %02d:%02d (分:秒)\n" $minutes $seconds
else
    printf "脚本运行时长: %d 秒\n" $duration
fi