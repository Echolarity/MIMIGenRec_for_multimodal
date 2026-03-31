import argparse
import os
import json
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ----------------- 基础工具函数 -----------------
def clean_text(text):
    if not text:
        return ""
    text = str(text).replace("\n", " ")
    return text.strip()

def load_json(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        return json.load(f)

# ----------------- 数据加载与融合 -----------------
def load_data(args):
    print(f"数据根目录: {args.root}")
    item2feature_path = os.path.join(args.root, f"{args.dataset}.item.json")
    item2feature = load_json(item2feature_path)
    return item2feature

def generate_multimodal_data(item2feature, text_features):
    item_data_list = []
    missing_img_cnt = 0
    
    for item_idx in item2feature:
        data = item2feature[item_idx]
        text_parts = []
        for meta_key in text_features:
            if meta_key in data and data[meta_key]:
                cleaned = clean_text(data[meta_key])
                if cleaned:
                    text_parts.append(cleaned)
        
        final_text = " ".join(text_parts) if text_parts else "unknown item"
        
        try:
            item_id = int(item_idx)
        except ValueError:
            item_id = item_idx

        image_paths = data.get("image_paths", [])
        valid_img_path = None
        if image_paths and os.path.exists(image_paths[0]):
            valid_img_path = image_paths[0]
        else:
            missing_img_cnt += 1

        item_data_list.append((item_id, final_text, valid_img_path))

    return item_data_list, missing_img_cnt

# ----------------- 🎯 核心推理提取逻辑 (已修复维度) -----------------
def generate_item_embedding(args, item_data_list, processor, model, accelerator):
    all_ids = [x[0] for x in item_data_list]
    all_texts = [x[1] for x in item_data_list]
    all_img_paths = [x[2] for x in item_data_list]

    total_items = len(all_texts)
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    chunk_size = int(np.ceil(total_items / num_processes))
    start_idx = process_index * chunk_size
    end_idx = min(start_idx + chunk_size, total_items)

    local_ids = all_ids[start_idx:end_idx]
    local_texts = all_texts[start_idx:end_idx]
    local_img_paths = all_img_paths[start_idx:end_idx]

    if accelerator.is_main_process:
        print(f"Total items: {total_items}")
        print(f"Start generating multimodal embeddings with {num_processes} processes...")

    local_results = []
    batch_size = args.batch_size

    pbar = tqdm(total=len(local_texts), desc=f"Proc {process_index}", disable=not accelerator.is_local_main_process)

    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 🔥 抽取局部前向推理函数，方便在 concat 模式下调用两次
    def _get_embeddings(messages, images):
        text_prompts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in messages
        ]
        batch_inputs = processor(
            text=text_prompts,
            images=images if images else None,
            padding=True,
            return_tensors="pt"
        ).to(accelerator.device)

        outputs = model(**batch_inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1] 
        attention_mask = batch_inputs.attention_mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return (sum_embeddings / sum_mask).to(torch.float32).cpu().numpy()

    with torch.no_grad():
        for i in range(0, len(local_texts), batch_size):
            batch_ids = local_ids[i : i + batch_size]
            batch_texts = local_texts[i : i + batch_size]
            batch_img_paths = local_img_paths[i : i + batch_size]

            multi_msgs, multi_imgs = [], []
            text_msgs = []
            img_msgs, img_only_imgs = [], []

            for txt, img_path in zip(batch_texts, batch_img_paths):
                safe_txt = txt[:1500] 
                
                # 图片加载逻辑
                img_obj, has_img = None, False
                if img_path:
                    try:
                        img_obj = Image.open(img_path).convert("RGB")
                        img_obj.thumbnail((384, 384), Image.Resampling.LANCZOS)
                        has_img = True
                    except Exception as e:
                        pass
                
                # 🔥 根据需求分装内容
                if args.mode == "multimodal":
                    content = []
                    if has_img:
                        content.append({"type": "image"})
                        multi_imgs.append(img_obj)
                    content.append({"type": "text", "text": safe_txt})
                    multi_msgs.append([{"role": "user", "content": content}])

                elif args.mode == "text_only":
                    multi_msgs.append([{"role": "user", "content": [{"type": "text", "text": safe_txt}]}])

                elif args.mode == "image_only":
                    content = []
                    if has_img:
                        content.append({"type": "image"})
                        multi_imgs.append(img_obj)
                    else:
                        content.append({"type": "text", "text": " "}) # 无图片时用空格占位防报错
                    multi_msgs.append([{"role": "user", "content": content}])
                
                elif args.mode == "concat":
                    # 1. 设置文本部分
                    text_msgs.append([{"role": "user", "content": [{"type": "text", "text": safe_txt}]}])
                    # 2. 设置图片部分
                    content_img = []
                    if has_img:
                        content_img.append({"type": "image"})
                        img_only_imgs.append(img_obj)
                    else:
                        content_img.append({"type": "text", "text": " "})
                    img_msgs.append([{"role": "user", "content": content_img}])

            # 🔥 根据模式推断
            if args.mode in ["multimodal", "text_only", "image_only"]:
                mean_output = _get_embeddings(multi_msgs, multi_imgs)
            elif args.mode == "concat":
                # 分开推两遍然后直接按维度拼接
                emb_text = _get_embeddings(text_msgs, [])
                emb_img = _get_embeddings(img_msgs, img_only_imgs)
                mean_output = np.concatenate([emb_text, emb_img], axis=1) # shape: (batch_size, hidden_dim * 2)

            for idx, emb in zip(batch_ids, mean_output):
                local_results.append((idx, emb))

            pbar.update(len(batch_texts))

    pbar.close()
    accelerator.wait_for_everyone()

    all_results_flat = gather_object(local_results)

    if accelerator.is_main_process:
        print("Gathering finished. Sorting and saving...")
        all_results_flat.sort(key=lambda x: x[0])
        final_embeddings = np.stack([x[1] for x in all_results_flat], axis=0)
        
        print("Final Embeddings shape: ", final_embeddings.shape)
        # 根据实际 mode 命名文件
        file_name = f"{args.dataset}.emb-{args.plm_name}-{args.mode}.npy"
        file_path = os.path.join(args.root, file_name)
        np.save(file_path, final_embeddings)
        print(f"✅ 特征已成功保存至: {file_path}")

# ----------------- 模型与环境初始化 -----------------
def load_qwen_vl_model(model_path, accelerator):
    if accelerator.is_main_process:
        print(f"Loading Qwen3-VL Processor & Model from: {model_path}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # 🔥 修复辅助：避免控制台报 torch_dtype deprecated 警告（新版transformers参数对齐）
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        dtype=torch.bfloat16,  # 规避警告
        low_cpu_mem_usage=True
    )
    return processor, model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Industrial_and_Scientific")
    parser.add_argument("--root", type=str, default="./processed_data")
    parser.add_argument("--plm_name", type=str, default="Qwen3VL")
    parser.add_argument("--plm_checkpoint", type=str, default="/root/sunyuqi/models/Qwen3-VL-8B-Instruct")
    # 🔥 添加实验模式控制参数
    parser.add_argument(
        "--mode", 
        type=str, 
        default="multimodal", 
        choices=["multimodal", "text_only", "image_only", "concat"],
        help="特征提取模式"
    )
    parser.add_argument("--max_sent_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4, help="推荐2或4")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        print(f"🚀 Running with {accelerator.num_processes} processes.")

    item2feature = load_data(args)
    item_data_list, missing_imgs = generate_multimodal_data(item2feature, ["title", "description"])
    
    if accelerator.is_main_process:
        print(f"Loaded {len(item_data_list)} items. Items without valid images: {missing_imgs}")

    plm_processor, plm_model = load_qwen_vl_model(args.plm_checkpoint, accelerator)
    
    plm_model = plm_model.to(accelerator.device)
    plm_model.eval()

    generate_item_embedding(args, item_data_list, plm_processor, plm_model, accelerator)