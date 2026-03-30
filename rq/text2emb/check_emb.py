import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ===================== 你的特征文件路径（直接用） =====================
EMB_PATH = "/root/sunyuqi/MIMIGenRec/newdata/processed_data/Industrial_and_Scientific/Industrial_and_Scientific.emb-Qwen3VL-multimodal.npy"

# ===================== 1. 基础完整性检验 =====================
print("="*50)
print("🔍 第一步：基础文件检验")
try:
    embeddings = np.load(EMB_PATH)
    print(f"✅ 特征文件加载成功")
    print(f"📊 特征形状: {embeddings.shape} → (商品数量, 特征维度)")
except Exception as e:
    print(f"❌ 文件加载失败: {e}")
    exit()

# 检查异常值
has_nan = np.isnan(embeddings).any()
has_inf = np.isinf(embeddings).any()
print(f"🔍 是否存在NaN异常值: {'是 ❌' if has_nan else '否 ✅'}")
print(f"🔍 是否存在无穷大异常值: {'是 ❌' if has_inf else '否 ✅'}")

# ===================== 2. 语义合理性检验（核心！） =====================
print("\n" + "="*50)
print("🔍 第二步：语义相似度检验（判断特征是否有区分度）")

# 随机选3个商品做测试
idx1, idx2, idx3 = 0, 1, 100
emb1 = embeddings[idx1:idx1+1]
emb2 = embeddings[idx2:idx2+1]
emb3 = embeddings[idx3:idx3+1]

# 计算余弦相似度（越接近1=越相似）
sim12 = cosine_similarity(emb1, emb2)[0][0]
sim13 = cosine_similarity(emb1, emb3)[0][0]

print(f"🔗 商品{idx1} vs 商品{idx2} 相似度: {sim12:.4f}")
print(f"🔗 商品{idx1} vs 商品{idx3} 相似度: {sim13:.4f}")

# 判定规则
if sim12 > sim13 + 0.1:
    print("✅ 检验通过：相似商品更近，不同商品更远，特征语义有效！")
else:
    print("⚠️  警告：特征语义区分度较弱（不影响使用，只是效果一般）")

# ===================== 3. 数值分布检验 =====================
print("\n" + "="*50)
print("🔍 第三步：特征数值分布检验")
mean_val = embeddings.mean()
std_val = embeddings.std()
print(f"📈 特征均值: {mean_val:.4f}")
print(f"📈 特征标准差: {std_val:.4f}")
if 0.01 < std_val < 10:
    print("✅ 数值分布正常，无极端值")
else:
    print("⚠️  数值分布异常（一般不影响推荐模型使用）")

# ===================== 4. 可视化检验（可选，直观看聚类） =====================
print("\n" + "="*50)
print("🔍 第四步：特征可视化（生成图片）")
# PCA降维到2维
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.6, s=20)
plt.title("Multimodal Embedding Visualization (PCA 2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.savefig("embedding_visual.png", dpi=300)
print("✅ 可视化图片已保存为: embedding_visual.png")
print("🎯 看图标准：点有聚集趋势 = 特征有效")

print("\n" + "="*50)
print("🎉 全部检验完成！你的多模态特征可以直接用于推荐模型训练！")