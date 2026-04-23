import argparse
import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入 ModelScope 的下载工具
from modelscope import snapshot_download

def extract_codebooks(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    codebooks = {}
    for k, v in state_dict.items():
        if "vq_layers" in k and v.ndim == 2:
            match = re.search(r'vq_layers\.(\d+)', k)
            if match:
                level = int(match.group(1))
                if level not in codebooks or v.numel() > codebooks[level].numel():
                    codebooks[level] = v
    return codebooks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--new_tokens_path", type=str, required=True)
    parser.add_argument("--rqvae_ckpt", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    # ========================== [新增: 控制先验软注入的超参] ==========================
    parser.add_argument("--blend_ratio", type=float, default=0.5, 
                        help="先验与随机初始化的融合比例(0.0纯随机~1.0纯先验)，推荐0.3-0.5避免过度压制序列因果")
    parser.add_argument("--noise_scale", type=float, default=0.01, 
                        help="为打破平铺带来的维度对称性而注入的高斯噪声标准差")
    # =================================================================================
    args = parser.parse_args()

    # ModelScope 上的 Qwen 组织名为小写 'qwen'，这里做一下自适应修复防呆
    ms_model_id = args.model_id
    if ms_model_id.startswith("Qwen/"):
        ms_model_id = ms_model_id.replace("Qwen/", "qwen/")

    print(f"[*] 当前使用 ModelScope 拉取基座模型: {ms_model_id}")
    # 这一步会从国内满速下载并返回本地缓存目录路径 (如果下过了会直接返回本地路径)
    model_dir = snapshot_download(ms_model_id)
    print(f"[*] 模型本地缓存路径: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        torch_dtype=torch.bfloat16, 
        device_map="cpu", 
        trust_remote_code=True
    )

    # ========================== [核心修改] 严谨确认隐空间大小 ==========================
    # 优先从模型配置中抓取底层设计的 Hidden Size
    config = model.config
    hidden_size = getattr(config, "hidden_size", getattr(config, "d_model", getattr(config, "dim", None)))

    # 获取实际权重张量，做双重对齐校验
    embeddings = model.get_input_embeddings().weight.data
    actual_tensor_dim = embeddings.shape[1]

    # 致命错误阻断：防止因加载错误模型而导致的错位计算
    if hidden_size is None:
        raise ValueError("❌ 无法从模型 Config 解析到参数 hidden_size / d_model。")
    if hidden_size != actual_tensor_dim:
        raise ValueError(f"❌ 严重维度张量错误: Config架构定义维度({hidden_size}) 与 实际底层权重张量({actual_tensor_dim}) 不匹配！")
    
    print(f"[*] 安全校验通过！已严谨确认模型 {args.model_id} 的隐层维度为: {hidden_size}")
    # =================================================================================

    with open(args.new_tokens_path, "r") as f:
        new_tokens = json.load(f)

    print(f"[*] Adding {len(new_tokens)} tokens to tokenizer...")
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    print(f"[*] Extracting Codebooks from {args.rqvae_ckpt}...")
    codebooks = extract_codebooks(args.rqvae_ckpt)
    
    # 重新获取 resize 之后的 Embedding (保证索引不会越界)
    embeddings = model.get_input_embeddings().weight.data
    orig_mean = embeddings.mean().item()
    orig_std = embeddings.std().item()

    prefix2level = {"<a_": 0, "<b_": 1, "<c_": 2, "<d_": 3, "<e_": 4}
    injected_count = 0
    
    # # 将 RQ-VAE 连续表征按规则赋予给新增的 Token ID
    # for token in new_tokens:
    #     token_id = tokenizer.convert_tokens_to_ids(token)
    #     prefix = token[:3]
    #     try:
    #         idx = int(token[3:-1])
    #     except ValueError:
    #         continue
            
    #     if prefix in prefix2level:
    #         level = prefix2level[prefix]
    #         if level in codebooks:
    #             # 获取该类别下，具体的表征向量（RQ-VAE Code）
    #             code_emb = codebooks[level][idx] 
                
    #             # Math: 平铺堆叠（Tiling）。例如：Code为32维，LLM隐空间为896维，32平铺28次恰好补满896
    #             repeats = hidden_size // code_emb.shape[0]
    #             rem = hidden_size % code_emb.shape[0]
    #             tiled = code_emb.repeat(repeats)
    #             if rem > 0:
    #                 tiled = torch.cat([tiled, code_emb[:rem]])
                
    #             # Normalization: 幅度分布对齐原生的LLM隐空间，防止摧毁原注意力机制 (Attention Mechanism)
    #             tiled = (tiled - tiled.mean()) / (tiled.std() + 1e-6)
    #             tiled = tiled * orig_std + orig_mean
                
    #             # 安全注入并覆写掉系统此前的随机初始化浮点
    #             embeddings[token_id] = tiled.to(embeddings.dtype)
    #             injected_count += 1
    # ========================== [核心修改: 对称性破局与软特征融合] ==========================
    # ======= [核心修复代码: 修复方差坍塌与信噪比问题] =======
    # 建议命令行参数恢复默认: --blend_ratio 0.5 --noise_scale 0.01
    
    for token in new_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        prefix = token[:3]
        try:
            idx = int(token[3:-1])
        except ValueError:
            continue
            
        if prefix in prefix2level:
            level = prefix2level[prefix]
            if level in codebooks:
                code_emb = codebooks[level][idx] 
                
                # 1. 空间补全: Math 平铺 (Tiling)
                repeats = hidden_size // code_emb.shape[0]
                rem = hidden_size % code_emb.shape[0]
                tiled = code_emb.repeat(repeats)
                if rem > 0:
                    tiled = torch.cat([tiled, code_emb[:rem]])
                
                # 2. 先标准化对齐，获取原信号 (让先验达到 LLM 的量级)
                tiled = (tiled - tiled.mean()) / (tiled.std() + 1e-6)
                tiled_rescaled = tiled * orig_std + orig_mean
                
                # 3. 注入微量噪声破局 (以 LLM 真实的 std 为尺度基准，极度安全)
                # 0.01 意味着我们只扰动原生方差 1% 的微小噪声来打破周期对称性
                noise = torch.randn_like(tiled_rescaled) * (orig_std * args.noise_scale)
                tiled_rescaled = tiled_rescaled + noise
                
                # 4. 软先验融合 (Blend)
                random_init = embeddings[token_id].clone()
                blended = args.blend_ratio * tiled_rescaled + (1.0 - args.blend_ratio) * random_init
                
                # 5. [最致命缺漏修复：恢复方差坍塌]！
                # 融合必定导致方差变小模长缩水，必须再重置回 LLM 的均值方差以激活注意力机制
                blended = (blended - blended.mean()) / (blended.std() + 1e-6)
                final_vec = blended * orig_std + orig_mean
                
                # 6. 回写
                embeddings[token_id] = final_vec.to(embeddings.dtype)
                injected_count += 1
    print(f"[*] 成功为 {injected_count} 个推荐 Token 注入了具备几何空间感知能力的视觉先验。")

    # 落地保存修改好的带视觉先验的模型
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print(f"[*] Saved prior-injected model to {args.output_path}")

if __name__ == "__main__":
    main()