import argparse
import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

Orthogonal_Loss_Weight=0.2

# ----------------- 🛠️ 新增：包含协同信息 (Collab) 的 MoE 特征解耦学习网络 -----------------
class MoECollabDecoupler(nn.Module):
    def __init__(self, input_dim, collab_dim):
        super().__init__()
        # 降维映射，因为 SASRec 的表征通常较小(128)，需先向上投影同文本/图像隐空间同维对齐
        self.collab_proj = nn.Sequential(
            nn.Linear(collab_dim, input_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(input_dim, input_dim)
        )
        # 三个独立主专家
        self.expert_text = nn.Sequential(nn.Linear(input_dim, input_dim), nn.LeakyReLU(0.1), nn.Linear(input_dim, input_dim))
        self.expert_image = nn.Sequential(nn.Linear(input_dim, input_dim), nn.LeakyReLU(0.1), nn.Linear(input_dim, input_dim))
        self.expert_collab = nn.Sequential(nn.Linear(input_dim, input_dim), nn.LeakyReLU(0.1), nn.Linear(input_dim, input_dim))
        
        # 共享信息专家
        self.expert_shared = nn.Sequential(nn.Linear(input_dim * 3, input_dim), nn.LeakyReLU(0.1), nn.Linear(input_dim, input_dim))
        # 门控机制接收三个模态做 Softmax 打分分配资源
        self.gate = nn.Sequential(nn.Linear(input_dim * 3, 4), nn.Softmax(dim=-1))
        
        # 重构回推解码器
        self.decoder_t = nn.Linear(input_dim, input_dim)
        self.decoder_i = nn.Linear(input_dim, input_dim)
        self.decoder_c = nn.Linear(input_dim, input_dim)

    def ortho_loss(self, f1, f2):
        return torch.mean((F.normalize(f1, p=2, dim=-1) * F.normalize(f2, p=2, dim=-1)).sum(dim=-1) ** 2)

    def forward(self, x_t, x_i, x_c):
        x_c_proj = self.collab_proj(x_c)

        h_t = self.expert_text(x_t)
        h_i = self.expert_image(x_i)
        h_c = self.expert_collab(x_c_proj)
        
        concat_input = torch.cat([x_t, x_i, x_c_proj], dim=-1)
        h_s = self.expert_shared(concat_input)
        
        loss_ortho = self.ortho_loss(h_t, h_s) + self.ortho_loss(h_i, h_s) + self.ortho_loss(h_c, h_s) + \
                     self.ortho_loss(h_t, h_i) + self.ortho_loss(h_t, h_c) + self.ortho_loss(h_i, h_c)
                     
        gate_scores = self.gate(concat_input)
        fused = (gate_scores[:, 0:1] * h_t) + (gate_scores[:, 1:2] * h_i) + \
                (gate_scores[:, 2:3] * h_c) + (gate_scores[:, 3:4] * h_s)
                
        loss_rec = F.mse_loss(self.decoder_t(fused), x_t) + \
                   F.mse_loss(self.decoder_i(fused), x_i) + \
                   F.mse_loss(self.decoder_c(fused), x_c_proj)
                   
        total_loss = loss_rec + Orthogonal_Loss_Weight * loss_ortho
        return fused, total_loss

# ----------------- 🛠️ 新增：MoE 特征解耦学习网络 -----------------
class MoEDecoupler(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.expert_text = nn.Sequential(nn.Linear(input_dim, input_dim), nn.LeakyReLU(0.1), nn.Linear(input_dim, input_dim))
        self.expert_image = nn.Sequential(nn.Linear(input_dim, input_dim), nn.LeakyReLU(0.1), nn.Linear(input_dim, input_dim))
        self.expert_shared = nn.Sequential(nn.Linear(input_dim * 2, input_dim), nn.LeakyReLU(0.1), nn.Linear(input_dim, input_dim))
        
        self.gate = nn.Sequential(nn.Linear(input_dim * 2, 3), nn.Softmax(dim=-1))
        
        self.decoder_t = nn.Linear(input_dim, input_dim)
        self.decoder_i = nn.Linear(input_dim, input_dim)

    def ortho_loss(self, f1, f2):
        f1_norm = F.normalize(f1, p=2, dim=-1)
        f2_norm = F.normalize(f2, p=2, dim=-1)
        return torch.mean((f1_norm * f2_norm).sum(dim=-1) ** 2)

    def forward(self, x_t, x_i):
        h_t = self.expert_text(x_t)
        h_i = self.expert_image(x_i)
        concat_input = torch.cat([x_t, x_i], dim=-1)
        h_s = self.expert_shared(concat_input)
        
        loss_ortho = self.ortho_loss(h_t, h_s) + self.ortho_loss(h_i, h_s) + self.ortho_loss(h_t, h_i)
                     
        gate_scores = self.gate(concat_input) 
        fused = (gate_scores[:, 0:1] * h_t) + (gate_scores[:, 1:2] * h_i) + (gate_scores[:, 2:3] * h_s)
                
        rec_t = self.decoder_t(fused)
        rec_i = self.decoder_i(fused)
        loss_rec = F.mse_loss(rec_t, x_t) + F.mse_loss(rec_i, x_i)
        
        total_loss = loss_rec + Orthogonal_Loss_Weight * loss_ortho
        return fused, total_loss

# ----------------- 🛠️ 新增：对比对齐适配器网络 -----------------
class ContrastiveAdapter(nn.Module):
    def __init__(self, input_dim, proj_dim=None):
        super().__init__()
        if proj_dim is None: proj_dim = input_dim
        self.text_proj = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.image_proj = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, text_features, image_features):
        z_t = self.text_proj(text_features)
        z_i = self.image_proj(image_features)
        
        z_t = F.normalize(z_t, p=2, dim=-1)
        z_i = F.normalize(z_i, p=2, dim=-1)
        
        t = torch.clamp(self.temperature, min=1e-4) 
        logits_per_image = (z_i @ z_t.T) / t
        logits_per_text = (z_t @ z_i.T) / t
        
        batch_size = z_t.size(0)
        labels = torch.arange(batch_size, device=z_t.device)
        
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        
        return (loss_i + loss_t) / 2.0

    @torch.no_grad()
    def get_fused_features(self, text_features, image_features):
        z_t = self.text_proj(text_features)
        z_i = self.image_proj(image_features)
        return z_t + z_i

def clean_text(text):
    if not text: return ""
    return str(text).replace("\n", " ").strip()

def load_json(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        return json.load(f)

def load_data(args):
    print(f"数据根目录: {args.root}")
    item2feature = load_json(os.path.join(args.root, f"{args.dataset}.item.json"))
    return item2feature

def generate_multimodal_data(item2feature, text_features):
    item_data_list = []
    missing_img_cnt = 0
    
    for item_idx in item2feature:
        data = item2feature[item_idx]
        text_parts = [clean_text(data[k]) for k in text_features if k in data and clean_text(data[k])]
        final_text = " ".join(text_parts) if text_parts else "unknown item"
        
        item_id = int(item_idx) if str(item_idx).isdigit() else item_idx
        valid_img_path = data.get("image_paths", [None])[0] if data.get("image_paths") and os.path.exists(data.get("image_paths")[0]) else None
        if not valid_img_path: missing_img_cnt += 1

        item_data_list.append((item_id, final_text, valid_img_path))
    return item_data_list, missing_img_cnt

# ----------------- 🎯 核心推理提取逻辑 -----------------
def generate_item_embedding(args, item_data_list, processor, model, accelerator, item2feature=None):
    collab_embeddings = None
    if args.mode in ["multimodal_collab", "concat_collab", "moe_collab"]:
        if not args.collab_emb_path or not os.path.exists(args.collab_emb_path):
            raise ValueError(f"当前模式 {args.mode} 未正确挂载协同特征!")
        collab_embeddings = np.load(args.collab_emb_path)
        if accelerator.is_main_process:
            print(f"✅ 成功挂载 SASRec 协同表征，Shape: {collab_embeddings.shape}")
            
    all_ids, all_texts, all_img_paths = [x[0] for x in item_data_list], [x[1] for x in item_data_list], [x[2] for x in item_data_list]

    total_items = len(all_texts)
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    chunk_size = int(np.ceil(total_items / num_processes))
    start_idx = process_index * chunk_size
    end_idx = min(start_idx + chunk_size, total_items)

    local_ids = all_ids[start_idx:end_idx]
    local_texts = all_texts[start_idx:end_idx]
    local_img_paths = all_img_paths[start_idx:end_idx]

    local_results = []
    batch_size = args.batch_size

    pbar = tqdm(total=len(local_texts), desc=f"Proc {process_index}", disable=not accelerator.is_local_main_process)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    def _get_embeddings(messages, images):
        text_prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
        batch_inputs = processor(text=text_prompts, images=images if images else None, padding=True, return_tensors="pt").to(accelerator.device)

        outputs = model(**batch_inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1] 
        attention_mask = batch_inputs.attention_mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return (sum_embeddings / sum_mask).to(torch.float32).cpu().numpy()

    with torch.no_grad():
        for i in range(0, len(local_texts), batch_size):
            batch_ids, batch_texts, batch_img_paths = local_ids[i:i + batch_size], local_texts[i:i + batch_size], local_img_paths[i:i + batch_size]
            multi_msgs, multi_imgs, text_msgs, img_msgs, img_only_imgs = [], [], [], [], []

            for txt, img_path in zip(batch_texts, batch_img_paths):
                safe_txt = txt[:1500] 
                img_obj, has_img = None, False
                if img_path:
                    try:
                        img_obj = Image.open(img_path).convert("RGB")
                        img_obj.thumbnail((384, 384), Image.Resampling.LANCZOS)
                        has_img = True
                    except Exception: pass
                
                text_msgs.append([{"role": "user", "content": [{"type": "text", "text": safe_txt}]}])
                content_img = [{"type": "image"}] if has_img else [{"type": "text", "text": " "}]
                img_msgs.append([{"role": "user", "content": content_img}])
                if has_img: img_only_imgs.append(img_obj)
                
                if args.mode in ["multimodal", "multimodal_collab"]:
                    content = [{"type": "image"}] if has_img else []
                    if has_img: multi_imgs.append(img_obj)
                    content.append({"type": "text", "text": safe_txt})
                    multi_msgs.append([{"role": "user", "content": content}])
                    
                elif args.mode == "text_only":
                    multi_msgs.append([{"role": "user", "content": [{"type": "text", "text": safe_txt}]}])
                    
                elif args.mode == "image_only":
                    multi_msgs.append([{"role": "user", "content": content_img}])
                    if has_img: multi_imgs.append(img_obj)
                    
                elif args.mode == "image2text":
                    content = []
                    if has_img:
                        content.append({"type": "image"})
                        multi_imgs.append(img_obj)
                        content.append({"type": "text", "text": "Describe this product image in detail to serve as its description without any extra words."})
                    else: content.append({"type": "text", "text": "Describe this product based solely on text: " + safe_txt})
                    multi_msgs.append([{"role": "user", "content": content}]) 
                    
                elif args.mode == "image2text_supp":
                    content = []
                    if has_img:
                        content.append({"type": "image"})
                        multi_imgs.append(img_obj)
                        prompt = (f"Original text: {safe_txt[:800]}\n"
                                  f"Observe the image carefully. Provide a supplementary description focusing ONLY on visual details "
                                  f"(e.g., specific colors, textures, patterns, shapes, design elements, materials) that are entirely MISSING "
                                  f"or NOT mentioned in the text above. Output only the short supplementary details without any prefix.")
                        content.append({"type": "text", "text": prompt})
                    else: content.append({"type": "text", "text": "No image. Reply 'EMPTY'."})
                    multi_msgs.append([{"role": "user", "content": content}])            

            # ======= 分模式推断 =======
            if args.mode in ["image2text", "image2text_supp"]:
                text_prompts_gen = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in multi_msgs]
                batch_inputs_gen = processor(text=text_prompts_gen, images=multi_imgs if multi_imgs else None, padding=True, return_tensors="pt").to(accelerator.device)
                generated_ids = model.generate(**batch_inputs_gen, max_new_tokens=150)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(batch_inputs_gen.input_ids, generated_ids)]
                batch_generated_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                
                text_msgs_remap, final_texts_cache = [], []  
                for idx_in_batch, (g_txt, orig_txt) in enumerate(zip(batch_generated_texts, batch_texts)):
                    has_img_flag = (batch_img_paths[idx_in_batch] != "")
                    if not has_img_flag: g_txt = "" 

                    if args.mode == "image2text":
                        final_combined_txt = f"[Original Meta]: {orig_txt} [Image Visual]: {g_txt}" if (args.combine_orig_text and g_txt.strip()) else (g_txt if g_txt.strip() else orig_txt)
                    elif args.mode == "image2text_supp":
                        final_combined_txt = f"{orig_txt} [Visual Supplement]: {g_txt}" if (has_img_flag and g_txt.strip()) else orig_txt
                    
                    final_texts_cache.append(final_combined_txt)
                    text_msgs_remap.append([{"role": "user", "content": [{"type": "text", "text": final_combined_txt}]}])
                
                mean_output = _get_embeddings(text_msgs_remap, [])
                for idx, emb, f_txt in zip(batch_ids, mean_output, final_texts_cache): local_results.append((idx, emb, f_txt))
            
            elif args.mode in ["moe_decouple", "concat", "contrastive_adapter", "concat_collab", "moe_collab"]:
                emb_text = _get_embeddings(text_msgs, [])
                emb_img = _get_embeddings(img_msgs, img_only_imgs)
                
                if args.mode in ["concat_collab", "moe_collab"]:
                    batch_collab = np.array([collab_embeddings[int(idx)] for idx in batch_ids]) 
                    if args.mode == "moe_collab":
                        mean_output = np.concatenate([emb_text, emb_img, batch_collab], axis=1)
                    else: 
                        mean_output = np.concatenate([args.text_weight * emb_text, (1.0 - args.text_weight) * emb_img, batch_collab], axis=1) if args.text_weight >= 0.0 else np.concatenate([emb_text, emb_img, batch_collab], axis=1)
                elif args.mode in ["moe_decouple", "contrastive_adapter"]:
                    mean_output = np.concatenate([emb_text, emb_img], axis=1)
                else: 
                    mean_output = np.concatenate([args.text_weight * emb_text, (1.0 - args.text_weight) * emb_img], axis=1) if args.text_weight >= 0.0 else np.concatenate([emb_text, emb_img], axis=1)
                for idx, emb in zip(batch_ids, mean_output): local_results.append((idx, emb, None))                
                
            elif args.mode == "multimodal_collab":
                emb_multi = _get_embeddings(multi_msgs, multi_imgs)
                batch_collab = np.array([collab_embeddings[int(idx)] for idx in batch_ids])
                mean_output = np.concatenate([emb_multi, batch_collab], axis=1)
                for idx, emb in zip(batch_ids, mean_output): local_results.append((idx, emb, None))
                
            elif args.text_weight >= 0.0 and args.mode == "multimodal":
                emb_text = _get_embeddings(text_msgs, [])
                emb_img = _get_embeddings(img_msgs, img_only_imgs)
                mean_output = (args.text_weight * emb_text) + ((1.0 - args.text_weight) * emb_img)
                for idx, emb in zip(batch_ids, mean_output): local_results.append((idx, emb, None))
                
            else:
                if args.mode in ["multimodal", "text_only", "image_only"]: mean_output = _get_embeddings(multi_msgs, multi_imgs)
                for idx, emb in zip(batch_ids, mean_output): local_results.append((idx, emb, None))
                
            pbar.update(len(batch_texts))

    pbar.close()
    accelerator.wait_for_everyone()

    all_results_flat = gather_object(local_results)
    
    # ======= [安全防护] 提取和保存操作仅在主进程执跑 =======
    if accelerator.is_main_process:
        all_results_flat.sort(key=lambda x: x[0])
        final_embeddings = np.stack([x[1] for x in all_results_flat], axis=0)

        # 🚀 此处为 MoE/ 对比融合学习部分代码（原封不动）
        if args.mode == "moe_decouple":
            single_dim = final_embeddings.shape[1] // 2
            moe_model = MoEDecoupler(input_dim=single_dim).to(accelerator.device)
            optimizer = torch.optim.Adam(moe_model.parameters(), lr=1e-3, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.moe_epochs)
            dataloader = torch.utils.data.DataLoader(torch.tensor(final_embeddings, dtype=torch.float32), batch_size=256, shuffle=True)
            moe_model.train()
            best_loss, best_model_weights = float('inf'), None
            
            for epoch in range(args.moe_epochs):
                total_loss = 0.0
                for batch_feats in dataloader:
                    batch_feats = batch_feats.to(accelerator.device)
                    x_t, x_i = batch_feats[:, :single_dim], batch_feats[:, single_dim:]
                    optimizer.zero_grad()
                    _, loss = moe_model(x_t, x_i)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(moe_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
                scheduler.step()
                avg_loss = total_loss / len(dataloader)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_weights = copy.deepcopy(moe_model.state_dict())
            moe_model.load_state_dict(best_model_weights)
            moe_model.eval()
            fused_list = []
            with torch.no_grad():
                for batch_feats in torch.utils.data.DataLoader(torch.tensor(final_embeddings, dtype=torch.float32), batch_size=512, shuffle=False):
                    batch_feats = batch_feats.to(accelerator.device)
                    fused_out, _ = moe_model(batch_feats[:, :single_dim], batch_feats[:, single_dim:])
                    fused_list.append(fused_out.cpu().numpy())
            final_embeddings = np.concatenate(fused_list, axis=0)
            
        elif args.mode == "moe_collab":
            total_dim = final_embeddings.shape[1]
            collab_dim = collab_embeddings.shape[1]
            single_dim = (total_dim - collab_dim) // 2
            moe_model = MoECollabDecoupler(input_dim=single_dim, collab_dim=collab_dim).to(accelerator.device)
            optimizer = torch.optim.Adam(moe_model.parameters(), lr=1e-3, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.moe_epochs)
            dataloader = torch.utils.data.DataLoader(torch.tensor(final_embeddings, dtype=torch.float32), batch_size=256, shuffle=True)
            moe_model.train()
            best_loss, best_model_weights = float('inf'), None
            for epoch in range(args.moe_epochs):
                total_loss = 0.0
                for batch_feats in dataloader:
                    batch_feats = batch_feats.to(accelerator.device)
                    x_t = batch_feats[:, :single_dim]
                    x_i = batch_feats[:, single_dim:single_dim*2]
                    x_c = batch_feats[:, single_dim*2:]
                    optimizer.zero_grad()
                    _, loss = moe_model(x_t, x_i, x_c)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(moe_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
                scheduler.step()
                if (total_loss / len(dataloader)) < best_loss:
                    best_loss = total_loss / len(dataloader)
                    best_model_weights = copy.deepcopy(moe_model.state_dict())
            moe_model.load_state_dict(best_model_weights)
            moe_model.eval()
            fused_list = []
            with torch.no_grad():
                for batch_feats in torch.utils.data.DataLoader(torch.tensor(final_embeddings, dtype=torch.float32), batch_size=512, shuffle=False):
                    batch_feats = batch_feats.to(accelerator.device)
                    fused_out, _ = moe_model(batch_feats[:, :single_dim], batch_feats[:, single_dim:single_dim*2], batch_feats[:, single_dim*2:])
                    fused_list.append(fused_out.cpu().numpy())
            final_embeddings = np.concatenate(fused_list, axis=0) 

        elif args.mode == "contrastive_adapter":
            single_dim = final_embeddings.shape[1] // 2
            adapter_model = ContrastiveAdapter(input_dim=single_dim, proj_dim=single_dim).to(accelerator.device)
            optimizer = torch.optim.Adam(adapter_model.parameters(), lr=1e-3, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.moe_epochs)
            dataloader = torch.utils.data.DataLoader(torch.tensor(final_embeddings, dtype=torch.float32), batch_size=512, shuffle=True)
            adapter_model.train()
            best_loss, best_model_weights = float('inf'), None
            for epoch in range(args.moe_epochs):
                total_loss = 0.0
                for batch_feats in dataloader:
                    batch_feats = batch_feats.to(accelerator.device)
                    x_t, x_i = batch_feats[:, :single_dim], batch_feats[:, single_dim:]
                    optimizer.zero_grad()
                    loss = adapter_model(x_t, x_i)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(adapter_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
                scheduler.step()
                if (total_loss / len(dataloader)) < best_loss:
                    best_loss = total_loss / len(dataloader)
                    best_model_weights = copy.deepcopy(adapter_model.state_dict())
            adapter_model.load_state_dict(best_model_weights)
            adapter_model.eval()
            fused_list = []
            with torch.no_grad():
                for batch_feats in torch.utils.data.DataLoader(torch.tensor(final_embeddings, dtype=torch.float32), batch_size=1024, shuffle=False):
                    batch_feats = batch_feats.to(accelerator.device)
                    fused_list.append(adapter_model.get_fused_features(batch_feats[:, :single_dim], batch_feats[:, single_dim:]).cpu().numpy())
            final_embeddings = np.concatenate(fused_list, axis=0)

        file_path = os.path.join(args.root, f"{args.dataset}.emb-{args.plm_name}-{args.mode}.npy")
        np.save(file_path, final_embeddings)
        print(f"✅ 特征已成功保存至: {file_path}")
        
        if args.mode in ["image2text", "image2text_supp"] and item2feature is not None:
            new_item2feature = {}
            for item in all_results_flat:
                idx = str(item[0])   
                if idx in item2feature:
                    orig_data = item2feature[idx].copy()
                    orig_data["title"], orig_data["description"] = "", str(item[2])
                    new_item2feature[idx] = orig_data
            new_json_path = os.path.join(args.root, f"{args.dataset}.item-{args.mode}.json")
            with open(new_json_path, "w", encoding="utf-8") as f:
                json.dump(new_item2feature, f, indent=2, ensure_ascii=False)
            print(f"✅ 构建专用文本底层副本成功，已导出至: {new_json_path}")

    # 🌟【最核心优化点1】必须通过此处屏障挂起剩余的多闲置显卡工作进程。
    # 因为主节点提取完合并特征后，要做大量的单机版 MoE模型训练 或 IO保存，这些费时操作是在单卡 (Process 0) 进行的。 
    # 如果不强迫其他子进程等待主进程，从节点的代码走完就会向 torchrun 请求销毁环境释放端口，导致主进程死锁或莫名其妙被无情 kill 掉。
    accelerator.wait_for_everyone()

# ----------------- 模型与环境初始化 -----------------
def load_qwen_vl_model(model_path, accelerator):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # 🌟【最核心优化点2】强制设置 device_map 到局部当前卡
    # 避免了原先 4张卡并发把几十 GB 的 8B 模型通通拉入 CPU 内存后再分发从而引起宿主机 CPU OOM 宕机的灾难。利用此方法能做到各 GPU 点对点按需显存加载。
    # 顺便修复了 Transformers 老版本的 `dtype` 入参被淘汰所产生的潜在回退，应当标准化为 `torch_dtype`。
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,  
        low_cpu_mem_usage=True,
        device_map={"": accelerator.local_process_index}
    )
    return processor, model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Industrial_and_Scientific")
    parser.add_argument("--root", type=str, default="./processed_data")
    parser.add_argument("--plm_name", type=str, default="Qwen3VL")
    parser.add_argument("--plm_checkpoint", type=str, default="/root/sunyuqi/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--mode", type=str, default="multimodal", choices=["multimodal", "text_only", "image_only", "concat", "image2text", "image2text_supp", "moe_decouple", "contrastive_adapter", "multimodal_collab", "concat_collab", "moe_collab"])
    parser.add_argument("--collab_emb_path", type=str, default="", help="协同特征npy路径")
    parser.add_argument("--text_weight", type=float, default=-1.0)
    parser.add_argument("--combine_orig_text", action="store_true")
    parser.add_argument("--moe_epochs", type=int, default=30)
    parser.add_argument("--max_sent_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
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
    
    # 【优化点移除】：模型在初始化时，`device_map` 已经将其放置于分配给该进程的合理卡内，这里若再写死 .to() 极易同 DDP 参数发生映射抢占冲突，故移除废代码。
    plm_model.eval()

    generate_item_embedding(args, item_data_list, plm_processor, plm_model, accelerator, item2feature)