# # 提前配置可见GPU
# export PYTHONUNBUFFERED=1
# # 2. 破除多卡底层通信死锁的三大魔法环境变量
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_BLOCKING_WAIT=1   # 如果NCCL报错，强制它打印出错误而不是默默卡死
# # 3. 指定你的可用显卡
# export CUDA_VISIBLE_DEVICES="0,1,3,5"
# # accelerate launch --num_processes 1 amazon_multimodal_emb.py \
# #     --dataset Industrial_and_Scientific \
# #     --batch_size 2

# -------- [Industrial_and_Scientific] 多模态 Embedding --------
CUDA_VISIBLE_DEVICES="3,4,5" python /root/sunyuqi/MIMIGenRec/rq/text2emb/multimodal_amazon_text2emb.py \
    --dataset Industrial_and_Scientific \
    --root /root/sunyuqi/MIMIGenRec/newdata/processed_data/Industrial_and_Scientific \
    --plm_checkpoint /root/sunyuqi/models/Qwen3-VL-8B-Instruct \
    --batch_size 1 \
    --max_sent_len 1024