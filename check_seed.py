import json
import os
import argparse

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠️ 读取文件失败 {path}: {e}")
        return None

def compare_list_data(data1, data2):
    """专门对比包含测试样本的列表形 JSON (如 final_result.json)"""
    # 提取 input 作为唯一标识键，映射对应的 predict
    # 加入默认值避免某些特殊样本没有 input 字段
    dict1 = {item.get("input", f"idx_{i}"): item.get("predict") for i, item in enumerate(data1)}
    dict2 = {item.get("input", f"idx_{i}"): item.get("predict") for i, item in enumerate(data2)}

    if len(dict1) != len(dict2):
        print(f"  ❌ 失败：样本总数不一致！ 目录1({len(dict1)}) vs 目录2({len(dict2)})")
        return False

    diff_count = 0
    total = len(dict1)
    
    for input_text, predict1 in dict1.items():
        predict2 = dict2.get(input_text, None)
        if predict1 != predict2:
            diff_count += 1
            if diff_count <= 2:  # 只打印前2个不一致的样本作为排错参考
                print(f"    [不一致样例 {diff_count}]")
                print(f"    - 输入: {input_text[:80]}...")
                print(f"    - 目录1 预测: {predict1}")
                print(f"    - 目录2 预测: {predict2}")

    if diff_count == 0:
        print(f"  ✅ 完美匹配！共 {total} 条推理数据 100% 绝对一致。")
        return True
    else:
        print(f"  ❌ 警告：在这 {total} 条数据中，有 {diff_count} 条 ({(diff_count/total)*100:.2f}%) 预测不一致！")
        return False

def compare_dict_data(data1, data2):
    """专门对比指标类的字典形 JSON (如 metrics.json)"""
    all_keys = set(data1.keys()).union(set(data2.keys()))
    diff_count = 0
    
    for k in sorted(list(all_keys)):
        val1 = data1.get(k)
        val2 = data2.get(k)
        if val1 != val2:
            diff_count += 1
            print(f"    [指标差异] {k}: 目录1({val1}) vs 目录2({val2})")
            
    if diff_count == 0:
        print(f"  ✅ 完美匹配！共有 {len(all_keys)} 项指标对齐。")
        return True
    else:
        print(f"  ❌ 指标存在差异！共有 {diff_count} 项对不上。")
        return False

def process_file_pair(file_name, path1, path2):
    print(f"\n📄 正在对比文件: {file_name}")
    data1 = load_json(path1)
    data2 = load_json(path2)
    
    if data1 is None or data2 is None:
        return False
        
    if isinstance(data1, list) and isinstance(data2, list):
        return compare_list_data(data1, data2)
    elif isinstance(data1, dict) and isinstance(data2, dict):
        return compare_dict_data(data1, data2)
    else:
        # 如果一个存了列表一个存了字典，或者其他奇怪的情况
        if data1 == data2:
            print("  ✅ 完美匹配！(基础对象比对一致)")
            return True
        else:
            print("  ❌ 警告：文件格式不一致或内容差异极大！")
            return False

def main():
    parser = argparse.ArgumentParser(description="全自动对比两个文件夹下属的所有同名 JSON 文件")
    parser.add_argument("dir1", type=str, help="第一个结果文件夹路径 (例如: run1_output/)")
    parser.add_argument("dir2", type=str, help="第二个结果文件夹路径 (例如: run2_output/)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.dir1) or not os.path.isdir(args.dir2):
        print("🚨 错误：提供的路径中有非有效文件夹！请检查路径。")
        return
        
    print(f"🔍 扫描文件夹:\n ➡️ {args.dir1}\n ➡️ {args.dir2}")
    
    # 收集两个目录下的所有 .json 文件
    files1 = {f for f in os.listdir(args.dir1) if f.endswith('.json')}
    files2 = {f for f in os.listdir(args.dir2) if f.endswith('.json')}
    
    common_files = sorted(list(files1.intersection(files2)))
    
    if not common_files:
        print("\n⚠️ 没有找到任何同名的 .json 文件用于对比！请确保两个文件夹内有相同生成的文件。")
        return
        
    print(f"🧩 共找到 {len(common_files)} 个同名 JSON 文件即将进行深度对比。")
    
    match_count = 0
    
    for f in common_files:
        path1 = os.path.join(args.dir1, f)
        path2 = os.path.join(args.dir2, f)
        is_match = process_file_pair(f, path1, path2)
        if is_match:
            match_count += 1
            
    print("\n"+"="*40)
    print(f"🎯 最终比对汇总：测试了 {len(common_files)} 个文件，其中 {match_count} 个完全一致。")
    if match_count == len(common_files):
        print("🎉 恭喜！你的 Seed 完全锁死，整个推理逻辑的随机性已经被彻底击毙！")
    else:
        print("🚨 还需要排查：目前仍然有部分文件未能复现一致的结果。")
    print("="*40)    

if __name__ == "__main__":
    main()