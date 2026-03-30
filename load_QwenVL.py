import os
# 核心修复1：强制禁用 DynamicCache（根治维度不匹配）
os.environ["TRANSFORMERS_NO_DYNAMIC_CACHE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import transformers
from PIL import Image

# 补丁1: 兼容BeamSearchScorer
try:
    from transformers.generation.beam_search import BeamSearchScorer
    transformers.BeamSearchScorer = BeamSearchScorer
except ImportError:
    transformers.BeamSearchScorer = object

from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_path = "/root/sunyuqi/QwenVL"
    image_path = "/root/sunyuqi/MIMIGenRec/newdata/Industrial_and_Scientific_images/B0B1LNCNWV_MAIN.jpg"

    print("正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("正在加载 模型 (请耐心等待)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).eval()

    # 核心修复2：强化输入预处理补丁，彻底清空无效缓存
    original_prepare = model.prepare_inputs_for_generation
    def custom_prepare_inputs(*args, **kwargs):
        inputs = original_prepare(*args, **kwargs)
        # 强制重置非法的past_key_values
        inputs["past_key_values"] = None
        return inputs
    model.prepare_inputs_for_generation = custom_prepare_inputs

    print("✅ 模型加载完毕！开始图文推理...\n" + "="*40)

    # 标准图文输入
    query = tokenizer.from_list_format([
        {'image': image_path},
        {'text': '\n这张图片是一张'},
    ])

    inputs = tokenizer(query, return_tensors='pt').to(model.device)

    print("🧠 模型生成中...")
    with torch.no_grad():
        pred = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # 🔥 核心修复3：强制禁用KV缓存（根治NoneType报错）
            use_cache=False,
        )

    # 解析结果
    input_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(pred[0][input_len:], skip_special_tokens=True)

    print("="*40)
    print(f"图片路径: {image_path}")
    print(f"模型输出: {response}")

if __name__ == "__main__":
    main()