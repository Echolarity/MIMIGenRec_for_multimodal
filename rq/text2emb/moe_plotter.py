# moe_plotter.py
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def plot_moe_training_curves(history, is_collab, mode_name="moe"):
    """
    history: dict 包含 'loss_total', 'loss_rec', 'loss_ortho', 'ortho_weight', 'gates' 的列表
    """
    # 1. 创建基于时间戳的日志文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.getcwd(), "logs", f"moe_train_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n📊 正在生成 MoE 训练可视化图表，保存路径: {save_dir}")

    epochs = range(1, len(history['loss_total']) + 1)
    
    # 获取全局好看的画图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # ------------------ 图 1: 三种 Loss 变化曲线 ------------------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['loss_total'], label='Total Loss (Dynamic)', color='#d62728', linewidth=2)
    plt.plot(epochs, history['loss_rec'], label='Reconstruction Loss (Raw)', color='#1f77b4', linestyle='--')
    plt.plot(epochs, history['loss_ortho'], label='Orthogonal Loss (Raw)', color='#2ca02c', linestyle='-.')
    plt.title(f"[{mode_name}] Model Losses over Epochs", fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "1_losses_curve.png"), dpi=300)
    plt.close()

    # ------------------ 图 2: 正交参数动态权重变化 ------------------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['ortho_weight'], label='Adaptive Orthogonal Weight', color='#ff7f0e', linewidth=2.5)
    plt.title(f"[{mode_name}] Adaptive Orthogonal Weight Parameter", fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Equivalent Weight Ratio', fontsize=12)
    plt.fill_between(epochs, history['ortho_weight'], alpha=0.2, color='#ff7f0e')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "2_orthogonal_weight_curve.png"), dpi=300)
    plt.close()

    # ------------------ 图 3: 各模态专家门控权重比重 ------------------
    plt.figure(figsize=(10, 6))
    gates_array = np.array(history['gates']) # shape: (epochs, num_gates)
    
    # 提取各列数据
    text_w = gates_array[:, 0]
    img_w = gates_array[:, 1]
    
    if is_collab:
        collab_w = gates_array[:, 2]
        shared_w = gates_array[:, 3:].sum(axis=1) # 剩余的都是 shared
        
        plt.plot(epochs, text_w, label='Text Expert', color='#1f77b4', linewidth=2)
        plt.plot(epochs, img_w, label='Image Expert', color='#ff7f0e', linewidth=2)
        plt.plot(epochs, collab_w, label='Collab Expert', color='#2ca02c', linewidth=2)
        plt.plot(epochs, shared_w, label='Shared Experts (Sum)', color='#9467bd', linewidth=2, linestyle='--')
    else:
        shared_w = gates_array[:, 2:].sum(axis=1)
        
        plt.plot(epochs, text_w, label='Text Expert', color='#1f77b4', linewidth=2)
        plt.plot(epochs, img_w, label='Image Expert', color='#ff7f0e', linewidth=2)
        plt.plot(epochs, shared_w, label='Shared Experts (Sum)', color='#9467bd', linewidth=2, linestyle='--')

    plt.title(f"[{mode_name}] Expert Routing Gate Distribution", fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Ratio (Softmax Output)', fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "3_gate_weights_curve.png"), dpi=300)
    plt.close()

    print(f"✅ 三张可视化图表已成功保存至目录！\n")