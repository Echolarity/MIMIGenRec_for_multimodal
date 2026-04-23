-------- [Industrial_and_Scientific] --------
python rq/rqvae.py \
      --data_path /root/sunyuqi/MIMIGenRec/newdata/processed_data/Industrial_and_Scientific/Industrial_and_Scientific.emb-Qwen3VL-multimodal.npy \
      --ckpt_dir /root/sunyuqi/MIMIGenRec/newdata/rq_output/Industrial_and_Scientific \
      --lr 1e-3 \
      --epochs 10000 \
      --batch_size 20480

# -------- [Office_Products] --------
# python rq/rqvae.py \
#       --data_path data/Amazon18/Office_Products/Office_Products.emb-qwen-td.npy \
#       --ckpt_dir ./output/Office_Products \
#       --lr 1e-3 \
#       --epochs 10000 \
#       --batch_size 20480

# # -------- [Toys_and_Games] --------
# python rq/rqvae.py \
#       --data_path data/Amazon18/Toys_and_Games/Toys_and_Games.emb-qwen-td.npy \
#       --ckpt_dir ./output/Toys_and_Games \
#       --lr 1e-3 \
#       --epochs 10000 \
#       --batch_size 20480
