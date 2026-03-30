set -euo pipefail


# ===================== 配置区域 =====================
CATEGORY="Industrial_and_Scientific"
POSSESS_DATA_OUTPATH="./data/Amazon" 
CUDA_VISIBLE="0,3,4" # 选emb时候的多卡
MONO_CUDA="0" # 选一个充足的单卡
SAMPLE_SIZE=5000
USER_K=2
ITEM_K=2



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
    blue "========================================"
    # 1. 数据处理阶段
    green "[阶段1] 数据处理..." 
    python ./data/possess_data.py \
        --category ${CATEGORY} \
        --sample_size ${SAMPLE_SIZE} \
        --user_k ${USER_K} \
        --item_k ${ITEM_K} \
        --output_path ${POSSESS_DATA_OUTPATH}
        # 检查输出目录
    ROOT="${POSSESS_DATA_OUTPATH}/${CATEGORY}"
    if [ ! -d "$ROOT" ]; then
        red "错误: 数据目录未创建: $ROOT"
        exit 1
    fi
    green "✓ 数据处理完成"
    # 2. 文本到嵌入阶段
    green "[阶段2] text2emb..."
    check_path "$TEXT2EMB_MODEL"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE} python ./rq/text2emb/multimodal_amazon_text2emb.py \
        --dataset ${CATEGORY} \
        --root ${ROOT} \
        --plm_checkpoint ${TEXT2EMB_MODEL} \
        --batch_size 1 \
        --max_sent_len 1024
        # 检查嵌入文件
    NPY_FILE="${ROOT}/${CATEGORY}.emb-${MODEL_NAME}-multimodal.npy"
    if [ ! -f "$NPY_FILE" ]; then
        red "错误: 嵌入文件未生成: $NPY_FILE"
        exit 1
    fi
    green "✓ 嵌入转换完成: $(du -h "$NPY_FILE" | cut -f1)"
    green "[阶段3] RQ-VAE 训练..."
    mkdir -p "./rq_output/${CATEGORY}"
    python rq/rqvae.py \
      --data_path ${NPY_FILE}\
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
    green "[阶段4] 生成索引..."
    python rq/generate_indices.py \
        --dataset ${CATEGORY} \
        --ckpt_path ./rq_output/${CATEGORY}/best_collision_model.pth \
        --output_dir ${POSSESS_DATA_OUTPATH}/${CATEGORY} \
        --device cuda:${MONO_CUDA}
    green "✓ 索引生成完成"
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
    green "✓ 所有处理步骤完成!"
    blue "输出目录: $OUTPUT_DIR"
    # 显示最终文件结构
    if [ -d "$OUTPUT_DIR" ]; then
        echo ""
        blue "生成的文件:"
        tree "$OUTPUT_DIR" -L 2 || ls -la "$OUTPUT_DIR/"
    fi
    python preprocess_data_sft_rl.py \
        --data_dir $POSSESS_DATA_OUTPATH \
        --category $CATEGORY \
        --output_dir $OUTPUT_DIR \
        --seq_sample $TASK4_SAMPLE \
        --seed $SEED \
        --data_source $CATEGORY
    bash sft.sh
    bash evaluate.sh
}

check_command python
check_command nvidia-smi  # 检查 GPU 可用性

# 执行主程序
main














