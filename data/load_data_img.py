import json
import os
import requests
from tqdm import tqdm
from typing import List, Dict, Tuple

# ===================== 🔥 核心可配置参数 =====================
category = "Industrial_and_Scientific"
# 小样本数量：修改这个数字即可控制数据规模（推荐100/500/1000）
SAMPLE_SIZE = 500
# ============================================================

# 路径配置（无需修改）
file = f"/root/sunyuqi/MIMIGenRec/newdata/{category}.jsonl"          # 评论数据
meta_file = f"/root/sunyuqi/MIMIGenRec/newdata/meta_{category}.jsonl"# 商品元数据
img_dir = f"/root/sunyuqi/MIMIGenRec/newdata/{category}_images"  # 小样本专属图片文件夹

def load_jsonl(file_path: str, max_lines: int = None) -> List[Dict]:
    """
    加载jsonl文件（支持限制读取行数，适配小样本）
    :param file_path: 文件路径
    :param max_lines: 最大读取行数（None=全量）
    """
    data = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 限制读取行数（小样本核心）
        if max_lines:
            lines = lines[:max_lines]
        
        for line in tqdm(lines, desc=f"loading {os.path.basename(file_path)}"):
            line = line.strip()
            if not line:
                continue
            try:
                d_data = json.loads(line)
                data.append(d_data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e} in line: {line}")
    return data

def download_product_images(meta_data: List[Dict], save_dir: str):
    """下载商品图片（仅下载小样本商品的图片，带去重）"""
    os.makedirs(save_dir, exist_ok=True)
    downloaded_urls = set()

    print(f"\n开始下载【{SAMPLE_SIZE}个商品】的图片（本地已存在自动跳过）...")
    for product in tqdm(meta_data, desc="下载图片"):
        parent_asin = product.get("parent_asin", "unknown_asin")
        images = product.get("images", [])
        if not images:
            continue

        for img_info in images:
            img_url = img_info.get("hi_res") or img_info.get("large") or img_info.get("thumb")
            variant = img_info.get("variant", "default")
            if not img_url or img_url in downloaded_urls:
                continue

            img_name = f"{parent_asin}_{variant}.jpg"
            img_path = os.path.join(save_dir, img_name)
            if os.path.exists(img_path):
                downloaded_urls.add(img_url)
                continue

            try:
                response = requests.get(img_url, timeout=10, stream=True)
                response.raise_for_status()
                with open(img_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded_urls.add(img_url)
            except Exception as e:
                print(f"下载失败 {img_url}：{str(e)}")

def build_relation_dataset(review_data: List[Dict], meta_data: List[Dict], img_dir: str) -> Tuple[Dict, Dict]:
    """
    构建【文本+元数据+图片】统一关联数据集（仅处理小样本）
    """
    # 1. 小样本商品元数据映射
    product_map: Dict[str, Dict] = {}
    sample_asins = set()  # 小样本商品ID集合（用于筛选评论）
    for meta in meta_data:
        parent_asin = meta["parent_asin"]
        sample_asins.add(parent_asin)
        product_map[parent_asin] = {
            "meta": meta,
            "reviews": [],
            "image_paths": []
        }

    # 2. 仅筛选【小样本商品】的评论（大幅减少数据量）
    filtered_reviews = [r for r in review_data if r["parent_asin"] in sample_asins]
    # 3. 按商品分组评论
    review_map: Dict[str, List[Dict]] = {}
    for review in filtered_reviews:
        parent_asin = review["parent_asin"]
        if parent_asin not in review_map:
            review_map[parent_asin] = []
        review_map[parent_asin].append(review)

    # 4. 绑定评论+图片路径
    for parent_asin, reviews in review_map.items():
        if parent_asin in product_map:
            product_map[parent_asin]["reviews"] = reviews

    for parent_asin, product in product_map.items():
        image_paths = []
        if os.path.exists(img_dir):
            for img_name in os.listdir(img_dir):
                if img_name.startswith(f"{parent_asin}_"):
                    image_paths.append(os.path.abspath(os.path.join(img_dir, img_name)))
        product["image_paths"] = image_paths

    print(f"✅ 小样本筛选完成：商品{len(product_map)}个 | 评论{len(filtered_reviews)}条")
    return product_map, review_map

if __name__ == "__main__":
    # ===================== 1. 加载小样本数据 =====================
    print("========== 加载【小样本】评论数据 ==========")
    # 评论不限制行数（后续会自动筛选）
    review_data = load_jsonl(file)

    print("\n========== 加载【小样本】商品元数据 ==========")
    # 🔥 核心：仅加载前 SAMPLE_SIZE 个商品
    meta_data = load_jsonl(meta_file, max_lines=SAMPLE_SIZE)

    # ===================== 2. 下载小样本图片 =====================
    download_product_images(meta_data, img_dir)
    print(f"\n图片下载完成！小样本专属路径：{os.path.abspath(img_dir)}")

    # ===================== 3. 构建小样本关联数据集 =====================
    print("\n========== 构建【文本+图片】统一关联数据 ==========")
    product_map, review_map = build_relation_dataset(review_data, meta_data, img_dir)

    # ===================== 演示：小样本数据使用 =====================
    print("\n========== 小样本数据演示 ==========")
    demo_asin = list(product_map.keys())[0]
    demo_product = product_map[demo_asin]
    print(f"商品ID: {demo_asin}")
    print(f"商品元数据: {demo_product['meta']['title']}")
    print(f"该商品评论数: {len(demo_product['reviews'])}")
    print(f"该商品图片路径: {demo_product['image_paths']}")
    print(demo_product)