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

    with torch.no_grad():
        for i in range(0, len(local_texts), batch_size):
            batch_ids = local_ids[i : i + batch_size]
            batch_texts = local_texts[i : i + batch_size]
            batch_img_paths = local_img_paths[i : i + batch_size]

            multi_messages_list = []
            pil_images_list = []

            for txt, img_path in zip(batch_texts, batch_img_paths):
                content = []
                if img_path:
                    try:
                        img = Image.open(img_path).convert("RGB")
                        
                        # 🔥 修复关键1：强制压缩物理图片分辨率！
                        # thumbnail 会保持原比例，将其最大边长限制为 448 像素。
                        # 这会将图片的 Token 数量从 ~1800 暴降至 ~200，既保留了语义，又省出极多显存！
                        img.thumbnail((384, 384), Image.Resampling.LANCZOS)
                        
                        content.append({"type": "image"})
                        pil_images_list.append(img)
                    except Exception as e:
                        print(f"图像加载失败: {e}")
                
                # 🔥 修复关键2：人工预截断文本字符
                # 避免超长商品描述（2000字符大约等于500预估Token）
                safe_txt = txt[:1500] 
                content.append({"type": "text", "text": safe_txt})
                multi_messages_list.append([{"role": "user", "content": content}])

            text_prompts = [
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                for msg in multi_messages_list
            ]

            # 🔥 修复关键3：关闭这里的 truncation，因为我们在前面已经把图片和文本都安全压缩了！
            batch_inputs = processor(
                text=text_prompts,
                images=pil_images_list if pil_images_list else None,
                padding=True,
                return_tensors="pt"
            ).to(accelerator.device)

            # 提取最后一层隐藏状态
            outputs = model(**batch_inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1] 
            
            # Mean Pooling
            attention_mask = batch_inputs.attention_mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            
            sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            
            mean_output = sum_embeddings / sum_mask  
            mean_output = mean_output.to(torch.float32).cpu().numpy()

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
        file_name = f"{args.dataset}.emb-{args.plm_name}-multimodal.npy"
        file_path = os.path.join(args.root, file_name)
        np.save(file_path, final_embeddings)
        print(f"✅ 多模态特征已成功保存至: {file_path}")

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
    # max_sent_len 不需要了，但保留用于接口兼容
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