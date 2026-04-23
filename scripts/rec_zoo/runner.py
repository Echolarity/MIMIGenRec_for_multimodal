import argparse
import copy
import json
import os
import random
import glob

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .caser import Caser
from .dataset import RecDataset
from .gru import GRU
from .sasrec import SASRec


def parse_args():
    p = argparse.ArgumentParser(description="Rec zoo: traditional seq rec unleashed!")
    p.add_argument("--data_dir", type=str, default="data/Industrial_and_Scientific")
    p.add_argument("--split", type=str, default="sft", choices=["rl", "sft"])
    p.add_argument("--seq_size", type=int, default=50)
    p.add_argument("--epoch", type=int, default=1000)
    # 【改动1】降低 Batch size，显著增加迭代步数，让优化器充分热身
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--hidden_factor", type=int, default=128)  # 增强隐藏层表达力
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--l2_decay", type=float, default=1e-4)
    p.add_argument("--dropout_rate", type=float, default=0.5)
    # 【改动2】默认 CE Loss，小数据下全局负采样对极大缩小误差帮助巨大
    p.add_argument("--loss_type", type=str, default="ce", choices=["bce", "ce"])
    p.add_argument("--model", type=str, default="SASRec", choices=["SASRec", "GRU", "Caser"])
    p.add_argument("--num_filters", type=int, default=16)
    p.add_argument("--filter_sizes", type=str, default="[2,3,4]")
    # 【改动3】放宽早停机制，避免模型局部波动直接退赛
    p.add_argument("--early_stop", type=int, default=50)
    p.add_argument("--eval_num", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="")
    p.add_argument("--result_json", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# =========================================================================
# 精纯版只吃传统的稠密 .inter 数据，无视大模型文本
# =========================================================================
def load_traditional_data(data_dir, seq_size):
    train_files = glob.glob(os.path.join(data_dir, "*.train.inter"))
    valid_files = glob.glob(os.path.join(data_dir, "*.valid.inter"))
    test_files  = glob.glob(os.path.join(data_dir, "*.test.inter"))
    
    if not train_files or not valid_files or not test_files:
        raise FileNotFoundError(f"❌ 找不到 .inter 数据源！请检查 {data_dir} 内部是否有通过 possess 导出的原文件。")

    def parse_inter(file_path):
        samples =[]
        max_id = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) != 3: continue
                _, hist_str, target_id = parts
                
                hist = [int(x) for x in hist_str.split()] if hist_str.strip() else[]
                target = int(target_id)
                samples.append((hist, target))
                max_id = max(max_id, target, *(hist if hist else[]))
        return samples, max_id

    train_raw, max_train = parse_inter(train_files[0])
    valid_raw, max_valid = parse_inter(valid_files[0])
    test_raw,  max_test  = parse_inter(test_files[0])
    
    item_num = max(max_train, max_valid, max_test) + 1
    pad_id = item_num
    
    def pad_seq(seq, limit, pad_val):
        seq = seq[-limit:]
        pad_len = limit - len(seq)
        return seq +[pad_val] * pad_len

    train_samples =[(pad_seq(hist, seq_size, pad_id), target) for hist, target in train_raw]
    valid_samples =[(pad_seq(hist, seq_size, pad_id), target) for hist, target in valid_raw]
    test_samples  =[(pad_seq(hist, seq_size, pad_id), target) for hist, target in test_raw]
    
    return train_samples, valid_samples, test_samples, item_num, pad_id


def evaluate_with_predictions(model, samples, device, topk, pad_id):
    model.eval()
    hit_all = [0.0] * len(topk)
    ndcg_all = [0.0] * len(topk)
    bs = 1024
    for i in range(0, len(samples), bs):
        batch = samples[i : i + bs]
        seq = torch.LongTensor([s[0] for s in batch]).to(device)
        len_seq = torch.LongTensor([max(1, sum(1 for x in s[0] if x != pad_id)) for s in batch]).to(device)
        target = torch.LongTensor([s[1] for s in batch]).to(device)
        with torch.no_grad():
            pred = model.forward_eval(seq, len_seq)
        
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
            
        rank_list = pred.shape[1] - 1 - torch.argsort(torch.argsort(pred, dim=1), dim=1)
        target_rank = torch.gather(rank_list, 1, target.view(-1, 1)).view(-1)
        ndcg_full = 1.0 / torch.log2(target_rank + 2)
        
        for j, k in enumerate(topk):
            mask = (target_rank < k).float()
            hit_all[j] += mask.sum().cpu().item()
            ndcg_all[j] += (ndcg_full * mask).sum().cpu().item()
            
    n = len(samples)
    hr_list =[hit_all[j] / n for j in range(len(topk))]
    ndcg_list =[ndcg_all[j] / n for j in range(len(topk))]
    return hr_list, ndcg_list


def main():
    args = parse_args()
    if not args.save_dir:
        data_name = os.path.basename(os.path.normpath(args.data_dir))
        args.save_dir = f"experiments/{args.model}_{data_name}_Traditional"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    setup_seed(args.seed)

    if args.result_json is None:
        args.result_json = os.path.join(args.save_dir, "result.json")

    # 装载数据
    train_samples, valid_samples, test_samples, item_num, pad_id = load_traditional_data(
        args.data_dir, seq_size=args.seq_size
    )

    seq_size = args.seq_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.model == "SASRec":
        model = SASRec(args.hidden_factor, item_num, seq_size, args.dropout_rate, device)
    elif args.model == "GRU":
        model = GRU(args.hidden_factor, item_num, seq_size)
    else:
        model = Caser(args.hidden_factor, item_num, seq_size, args.num_filters, args.filter_sizes, args.dropout_rate)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    criterion = nn.BCEWithLogitsLoss() if args.loss_type == "bce" else nn.CrossEntropyLoss()
    model.to(device)
    
    train_dataset = RecDataset(train_samples, pad_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    topk =[1, 3, 5, 10, 20, 50]
    num_batches = len(train_loader)
    ndcg_max, best_epoch, early_stop = 0.0, 0, 0
    best_model = None

    for epoch in range(args.epoch):
        model.train()
        for batch_idx, (seq, len_seq, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
            seq = seq.to(device)
            len_seq = len_seq.to(device)
            target = target.to(device)

            if args.model == "GRU":
                len_seq = len_seq.cpu()

            optimizer.zero_grad()
            out = model.forward(seq, len_seq)

            if args.loss_type == "bce":
                target_neg =[]
                for t in target:
                    neg = np.random.randint(item_num)
                    while neg == t.item(): neg = np.random.randint(item_num)
                    target_neg.append(neg)
                target_neg = torch.LongTensor(target_neg).to(device)
                
                pos_scores = torch.gather(out, 1, target.view(-1, 1))
                neg_scores = torch.gather(out, 1, target_neg.view(-1, 1))
                scores = torch.cat((pos_scores, neg_scores), 0)
                labels = torch.cat((torch.ones_like(pos_scores), torch.zeros_like(neg_scores)), 0)
                loss = criterion(scores, labels)
            else:
                # 【极其强大的 CE 全局打分机制】，每次拉起 1 个正样本打压全部 N-1 个负样本
                loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            # 验证评估
            if (batch_idx + 1) % max(1, num_batches * args.eval_num) == 0 or (batch_idx + 1) == num_batches:
                val_hr, val_ndcg = evaluate_with_predictions(model, valid_samples, device, topk, pad_id)
                # 使用 NDCG@10 (index 3) 或 NDCG@20 (index 4) 作为监视标准
                ndcg_last = val_ndcg[4]  
                
                if ndcg_last > ndcg_max:
                    ndcg_max = ndcg_last
                    best_epoch = epoch
                    early_stop = 0
                    best_model = copy.deepcopy(model)
                else:
                    early_stop += 1
                
        # Epoch级别提前结束
        if early_stop >= args.early_stop:
            print(f">>> 稳定未涨幅，由 Early stooping (容忍度:{args.early_stop}) 提前停止于 Epoch {epoch}")
            break

    # 回滚最优模型并出分
    if best_model is None:
        best_model = model
        best_epoch = args.epoch - 1

    hr_tsv, ndcg_tsv = evaluate_with_predictions(best_model, test_samples, device, topk, pad_id)

    # 导出文件体系
    os.makedirs(os.path.dirname(args.result_json) or ".", exist_ok=True)
    result = {"best_epoch": best_epoch, "best_val_NDCG20": ndcg_max}
    with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "best_state.pth")
    torch.save(best_model.state_dict(), ckpt_path)
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "best_state.pth")
    torch.save(best_model.state_dict(), ckpt_path)

    # === 🔥 新增：自动安全提取并保存协同信息 Embedding 到 npy (完全解耦提取) ===
    try:
        for key, val in best_model.state_dict().items():
            # 找到参数维度符合 (item_num, hidden_factor) 且带 emb 或 weight 的层提取
            if len(val.shape) == 2 and val.shape[0] >= item_num and val.shape[1] == args.hidden_factor:
                emb_npy_path = os.path.join(args.save_dir, "sasrec_item_emb.npy")
                np.save(emb_npy_path, val.cpu().numpy())
                print(f"✅ 成功找到并提取协同 Embedding ({key}, shape: {val.shape})，保存在: {emb_npy_path}")
                break
    except Exception as e:
        print(f"⚠️ 无法自动提取协同 Embedding, 请检查模型内部定义: {e}")

    # ========================== 指定的控制台垂直打印样式 ==========================
    print("\n" + "=" * 45)
    for i, k in enumerate(topk):
        print(f"  NDCG@{k}: {ndcg_tsv[i]:.4f}")
    for i, k in enumerate(topk):
        print(f"  HR@{k}: {hr_tsv[i]:.4f}")
    print("=" * 45)
    print(f"✅ 测评完成！最优权重来自 Epoch {best_epoch}\n")

if __name__ == "__main__":
    main()