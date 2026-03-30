import argparse
import collections
import html
import json
import os
import re
import requests
from tqdm import tqdm
from typing import List, Dict, Tuple


# ===================== 🍉 1. 基础辅助函数 =====================
def clean_text(text):
    """清理文本：移除HTML标签、解码字符、去除多余空格 (同 amazon18)"""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", str(text))
    text = html.unescape(text)
    text = text.replace("&quot;", '"').replace("&amp;", "&")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def check_path(path):
    os.makedirs(path, exist_ok=True)

def write_json_file(data, file_path):
    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def write_remap_index(index_map, file_path):
    with open(file_path, "w", encoding='utf-8') as f:
        for original, mapped in index_map.items():
            f.write(f"{original}\t{mapped}\n")

# ===================== 🖼️ 2. 图像与数据加载 =====================
def download_product_images(meta_data: List[Dict], save_dir: str):
    """自动下载图片 (同原代码逻辑)"""
    os.makedirs(save_dir, exist_ok=True)
    downloaded_urls = set()
    for product in tqdm(meta_data, desc="Downloading images"):
        parent_asin = product.get("parent_asin", "unknown_asin")
        for img_info in product.get("images", []):
            img_url = img_info.get("hi_res") or img_info.get("large") or img_info.get("thumb")
            variant = img_info.get("variant", "default")
            # print(variant)
            if "MAIN" not in variant:
                # print(f"ignoring {variant}")
                continue
            if not img_url or img_url in downloaded_urls: continue
            
            img_path = os.path.join(save_dir, f"{parent_asin}_{variant}.jpg")
            if os.path.exists(img_path):
                downloaded_urls.add(img_url)
                continue
            try:
                response = requests.get(img_url, timeout=5, stream=True)
                response.raise_for_status()
                with open(img_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
                downloaded_urls.add(img_url)
            except: pass

def load_metadata_and_images(meta_file, img_dir, max_lines=None):
    """加载 Metadata，并挂载本地的 MAIN 图片，过滤掉没有 title 的项 (同 amazon18)"""
    metadata = []
    try:
        with open(meta_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if max_lines: lines = lines[:max_lines]
            for line in tqdm(lines, desc="Loading & Processing Metadata"):
                if not line.strip(): continue
                metadata.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"Metadata file {meta_file} not found")
        return [], {}, set()

    id_title = {}
    remove_items = set()
    processed_metadata = []

    for meta in tqdm(metadata, desc="Filtering Metadata & Attaching Images"):
        asin = meta.get("parent_asin")
        title = meta.get("title", "")
        
        # amazon18: 对 title 进行验证和过滤
        if not title or "<span id" in title:
            remove_items.add(asin)
            continue
            
        title = clean_text(title)
        
        # 🌟【关键图片筛选逻辑】：扫描文件夹，仅保留包含 MAIN 的图片
        image_paths = []
        if os.path.exists(img_dir):
            for img_name in os.listdir(img_dir):
                if img_name.startswith(f"{asin}_") and "MAIN" in img_name:
                    image_paths.append(os.path.abspath(os.path.join(img_dir, img_name)))
        
        meta["title"] = title
        meta["image_paths"] = image_paths
        
        if len(title) > 1 and len(title.split(" ")) <= 50:  # 放宽新版亚马逊标题长度限制
            id_title[asin] = title
            processed_metadata.append(meta)
        else:
            remove_items.add(asin)

    return processed_metadata, id_title, remove_items

def load_reviews(reviews_file, sample_asins):
    """仅加载在 Metadata (小样本) 中存在的 Reviews，统一时间格式"""
    reviews = []
    with open(reviews_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading Reviews"):
            r = json.loads(line.strip())
            # 新数据集字段对齐旧规范 format
            asin = r.get("parent_asin")
            if asin in sample_asins:
                r['unixReviewTime'] = int(r.get('timestamp', 0) / 1000) # ms -> s
                r['reviewerID'] = r.get("user_id")
                r['asin'] = asin
                r['overall'] = r.get("rating")
                r['reviewText'] = r.get("text", "")
                r['summary'] = r.get("title", "")
                reviews.append(r)
    return reviews


# ===================== ⚙️ 3. 核心过滤与转换 (Strict Amazon18 Style) =====================
def k_core_filtering(reviews, id_title, K=5):
    """严格的多次迭代 K-core 过滤"""
    remove_users = set()
    remove_items = set()

    for review in reviews:
        if review["asin"] not in id_title:
            remove_items.add(review["asin"])

    while True:
        new_reviews = []
        flag = False
        user_counts = collections.defaultdict(int)
        item_counts = collections.defaultdict(int)

        for review in reviews:
            if review["reviewerID"] in remove_users or review["asin"] in remove_items:
                continue
            user_counts[review["reviewerID"]] += 1
            item_counts[review["asin"]] += 1
            new_reviews.append(review)

        for user, count in user_counts.items():
            if count < K:
                remove_users.add(user)
                flag = True
        for item, count in item_counts.items():
            if count < K:
                remove_items.add(item)
                flag = True

        if not flag: break
        reviews = new_reviews

    print(f"K-core后统计: Users: {len(user_counts)}, Items: {len(item_counts)}, Reviews: {len(new_reviews)}")
    return new_reviews, user_counts, item_counts

def convert_inters2dict(reviews):
    """转换为 ID 并且针对用户进行时序排序"""
    user2index, item2index = dict(), dict()
    user_reviews = collections.defaultdict(list)
    
    for review in reviews:
        user_reviews[review["reviewerID"]].append(review)

    for user in user_reviews:
        user_reviews[user].sort(key=lambda x: int(x["unixReviewTime"]))

    for user in user_reviews:
        if user not in user2index: user2index[user] = len(user2index)
        for review in user_reviews[user]:
            item = review["asin"]
            if item not in item2index: item2index[item] = len(item2index)

    return user2index, item2index, user_reviews

def generate_interaction_list(user_reviews, user2index, item2index):
    """序列构造与时序记录生成 (亚马逊原版的滑动窗口思想)"""
    interaction_list = []
    
    for user_original, records in tqdm(user_reviews.items(), desc="Creating interaction sequences"):
        user_id = user2index[user_original]
        item_ids = [item2index[r["asin"]] for r in records]
        timestamps = [r["unixReviewTime"] for r in records]
        
        # 至少有一个历史和一个预测
        for i in range(1, len(item_ids)):
            st = max(i - 50, 0) # 限制最长历史为 50
            interaction_list.append([
                user_id,             # user index
                item_ids[st:i],      # history item indices
                item_ids[i],         # target item index
                timestamps[i]        # target timestamp (用于全局排序)
            ])
            
    # 根据目标商品的时间戳进行全局排序
    interaction_list.sort(key=lambda x: int(x[-1]))
    return interaction_list

def convert_to_atomic_files(interaction_list, dataset_name, out_dir):
    """8:1:1 划分并生成 txt (.inter) 格式数据集"""
    total_len = len(interaction_list)
    train_end = int(total_len * 0.8)
    valid_end = int(total_len * 0.9)

    train_data = interaction_list[:train_end]
    valid_data = interaction_list[train_end:valid_end]
    test_data = interaction_list[valid_end:]

    for name, data in zip(['train', 'valid', 'test'], [train_data, valid_data, test_data]):
        file_path = os.path.join(out_dir, f"{dataset_name}.{name}.inter")
        with open(file_path, "w") as f:
            f.write("user_id:token\titem_id_list:token_seq\titem_id:token\n")
            for inter in data:
                u = inter[0]
                hist = " ".join(map(str, inter[1]))
                target = inter[2]
                f.write(f"{u}\t{hist}\t{target}\n")
                
    return train_data, valid_data, test_data


# ===================== 🛠️ 4. 生成特性文件 (Feature JSONs) =====================
def create_item_features(metadata, item2index):
    """按规范抽取字典 (带有经过筛选的 image_paths)"""
    item2feature = {}
    asin_to_meta = {m['parent_asin']: m for m in metadata}

    for item_asin, item_id in item2index.items():
        if item_asin in asin_to_meta:
            meta = asin_to_meta[item_asin]
            
            # Categories处理：适配新旧结构
            cats = meta.get("categories", [])
            if isinstance(cats, list):
                if len(cats) > 0 and isinstance(cats[0], list):
                    flat_cats = [c for sub in cats for c in sub]
                    categories = ",".join([str(c).strip() for c in flat_cats])
                else:
                    categories = ",".join([str(c).strip() for c in cats])
            else:
                categories = str(cats)

            item2feature[item_id] = {
                "title": clean_text(meta.get("title", "")),
                "description": clean_text(' '.join(meta.get("description", [])) if isinstance(meta.get("description"), list) else meta.get("description", "")),
                "brand": meta.get("store", meta.get("brand", "")),
                "categories": categories.strip(),
                "image_paths": meta.get("image_paths", [])
            }
    return item2feature

def create_review_features(reviews, user2index, item2index):
    """构建独立评价特征库"""
    review_data = {}
    for r in tqdm(reviews, desc="Load reviews feature"):
        user, item = r["reviewerID"], r["asin"]
        if user in user2index and item in item2index:
            uid = user2index[user]
            iid = item2index[item]
            timestamp = r["unixReviewTime"]
            
            # 严格使用 amazon18_data_process 的索引格式拼接规则
            unique_key = f"{uid}_{iid}_{timestamp}"
            
            review_data[unique_key] = {
                "review": clean_text(r.get("reviewText", "")),
                "summary": clean_text(r.get("summary", ""))
            }
    return review_data

def parse_args():
    parser=argparse.ArgumentParser(
        description="数据处理参数配置",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--category', type=str, default='Industrial_and_Scientific',help='商品类别名称')
    parser.add_argument('--sample_size', type=int, default=500,
                       help='小样本商品数量，设置为0或负数时处理全量数据')
    parser.add_argument('--user_k', type=int, default=1,help='User K-core 过滤阈值')
    parser.add_argument('--item_k', type=int, default=1,
                       help='Item K-core 过滤阈值')
    parser.add_argument('--output_path', type=str, default='./processed_data',
                       help='处理后的数据输出路径')
    args=parser.parse_args()
    if args.sample_size <= 0:
        args.sample_size = None
    return args
def print_args(args):
    """打印所有参数"""
    print("=" * 60)
    print("参数配置:")
    print("=" * 60)
    for key, value in sorted(vars(args).items()):
        print(f"{key:20}: {value}")
    print("=" * 60)

# ===================== 🚀 5. 主程序流程 =====================
if __name__ == "__main__":
    args=parse_args()
    print(args)
    




    CATEGORY = args.category
    SAMPLE_SIZE = args.sample_size  # 小样本商品数量（设置 None 则处理全量）
    USER_K = args.user_k         # User K-core 过滤阈值
    ITEM_K = args.item_k        # Item K-core 过滤阈值
    OUTPUT_PATH =args.output_path

    # 原始数据路径配置
    FILE_REVIEWS = f"/root/sunyuqi/MIMIGenRec/data/{CATEGORY}.jsonl"
    FILE_META = f"/root/sunyuqi/MIMIGenRec/data/meta_{CATEGORY}.jsonl"
    IMG_DIR = f"/root/sunyuqi/MIMIGenRec/data/{CATEGORY}_images"

    out_dir = os.path.join(OUTPUT_PATH, CATEGORY)
    check_path(out_dir)

    print(f"========== 1. Metadata 加载与图像筛选 ==========")
    metadata, id_title, remove_items = load_metadata_and_images(
        FILE_META, IMG_DIR, max_lines=SAMPLE_SIZE
    )
    
    print(f"\n========== 2. [可选]自动下载缺失图像 ==========")
    download_product_images(metadata, IMG_DIR) # 依照需求开启，通常确保数据已在本地即可
    
    print(f"\n========== 3. 加载关联 Reviews ==========")
    sample_asins = set(id_title.keys())
    raw_reviews = load_reviews(FILE_REVIEWS, sample_asins)
    
    print(f"\n========== 4. K-Core 迭代过滤 ==========")
    f_reviews, user_cnt, item_cnt = k_core_filtering(raw_reviews, id_title, K=USER_K)
    
    print(f"\n========== 5. 数据转换与序列划分 (Strict 8:1:1) ==========")
    user2index, item2index, user_reviews = convert_inters2dict(f_reviews)
    interaction_list = generate_interaction_list(user_reviews, user2index, item2index)
    
    print(f"Total valid interaction sequences generated: {len(interaction_list)}")
    train, valid, test = convert_to_atomic_files(interaction_list, CATEGORY, out_dir)
    print(f"Split completed. Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")

    print(f"\n========== 6. 生成特征与映射文件 ==========")
    item_features = create_item_features(metadata, item2index)
    review_features = create_review_features(f_reviews, user2index, item2index)

    # 按照 amazon__data_process 保存格式
    write_json_file(item_features, os.path.join(out_dir, f"{CATEGORY}.item.json"))
    write_json_file(review_features, os.path.join(out_dir, f"{CATEGORY}.review.json"))
    write_remap_index(user2index, os.path.join(out_dir, f"{CATEGORY}.user2id"))
    write_remap_index(item2index, os.path.join(out_dir, f"{CATEGORY}.item2id"))

    print(f"\n✅ 全部处理完成！")
    print(f"📌 {CATEGORY}.item.json 已包含筛选后的 'MAIN' 图片绝对路径。")
    print(f"📁 结果保存在目录: {os.path.abspath(out_dir)}")