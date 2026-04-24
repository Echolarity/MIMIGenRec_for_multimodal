# moe_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUMiniLLMExpert(nn.Module):
    """
    【语言模型级别的专家模块】
    从 Llama / Qwen 等现代大型语言模型中提取出的核心层结构 (SwiGLU + LayerNorm 组合)。
    充当拥有极强非线性表征能力的微型 LLM 专家，相比普通的线性映射具备更深度的模态信息理解能力。
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        # 典型的语言模型门控前馈网络结构
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 残差和动态门控激活机制 (SwiGLU)
        residual = x
        x = self.norm1(x)
        # Swish(x) * x = x * sigmoid(beta * x) -> SILU等效
        swish_out = F.silu(self.fc1(x))
        gate_out = self.fc2(x)
        h = swish_out * gate_out
        
        out = self.dropout(self.fc3(h))
        return self.norm2(residual + out)


class MoEDecoupler(nn.Module):
    """
    【双模态与三模态统一的 MoE 动态网络】
    逻辑大一统：自动检测是否传入协同维度 (collab_dim)，从而动调决定启用双路还是三路专网，无需冗余代码。
    """
    def __init__(self, input_dim, collab_dim=None, num_shared_experts=2):
        super().__init__()
        self.input_dim = input_dim
        self.has_collab = collab_dim is not None
        
        # [动态结构] 模态对齐投影
        if self.has_collab:
            self.collab_proj = nn.Sequential(
                nn.Linear(collab_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.GELU(),
                nn.Linear(input_dim, input_dim)
            )
            concat_dim = input_dim * 3
            self.num_experts = 3 + num_shared_experts # 3个专属 + N个共享
        else:
            self.collab_proj = None
            concat_dim = input_dim * 2
            self.num_experts = 2 + num_shared_experts # 2个专属 + N个共享

        # [专家注册] 实例化 “微型语言模型” 作为各个模态打工的专属专家
        self.expert_text = SwiGLUMiniLLMExpert(input_dim, input_dim)
        self.expert_image = SwiGLUMiniLLMExpert(input_dim, input_dim)
        if self.has_collab:
            self.expert_collab = SwiGLUMiniLLMExpert(input_dim, input_dim)
            
        # 建立深度共享专家组
        self.shared_experts = nn.ModuleList([
            SwiGLUMiniLLMExpert(concat_dim if i==0 else input_dim, input_dim) 
            for i in range(num_shared_experts)
        ])
        
        # [动态门控] 加入温度调控的 Softmax 路由门，根据模态复杂性分配关注度
        self.gate = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 2),
            nn.LayerNorm(concat_dim // 2),
            nn.GELU(),
            nn.Linear(concat_dim // 2, self.num_experts)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        
        # [回推校验] 保真解码器
        self.decoder_t = nn.Linear(input_dim, input_dim)
        self.decoder_i = nn.Linear(input_dim, input_dim)
        if self.has_collab:
            self.decoder_c = nn.Linear(input_dim, input_dim)
            
        # self.Orthogonal_Loss_Weight = 0.2
        self.log_var_rec = nn.Parameter(torch.zeros(1))
        self.log_var_ortho = nn.Parameter(torch.zeros(1))

    def ortho_loss(self, f1, f2):
        """ 余弦相似度正交惩罚：迫使各专家的隐藏激活空间彼此独立 """
        f1_norm = F.normalize(f1, p=2, dim=-1)
        f2_norm = F.normalize(f2, p=2, dim=-1)
        return torch.mean((f1_norm * f2_norm).sum(dim=-1) ** 2)

    def forward(self, x_t, x_i, x_c=None):
        if self.has_collab and x_c is None:
            raise ValueError("初始化启用了三模态，但是未传入第三个模态特征。")
            
        # 1. 对齐与拼接
        if self.has_collab:
            x_c_proj = self.collab_proj(x_c)
            concat_input = torch.cat([x_t, x_i, x_c_proj], dim=-1)
        else:
            concat_input = torch.cat([x_t, x_i], dim=-1)

        # 2. 专属 LLM 专家前向提取
        h_t = self.expert_text(x_t)
        h_i = self.expert_image(x_i)
        if self.has_collab:
            h_c = self.expert_collab(x_c_proj)

        # 3. 共享 LLM 专家降维处理 (级联加深)
        s_out = self.shared_experts[0](concat_input)
        # 将共享专家从原始拼接维度降成 input_dim 后复用剩余专家
        if concat_input.shape[-1] != self.input_dim:
            s_out = s_out[:, :self.input_dim] 
            
        for i in range(1, len(self.shared_experts)):
            s_out = self.shared_experts[i](s_out)
        h_s = s_out 

        # 4. 计算门控得分
        temp = torch.clamp(self.temperature, min=0.01, max=2.0)
        gate_logits = self.gate(concat_input) / temp
        gate_scores = F.softmax(gate_logits, dim=-1)

        self._monitor_weights = gate_scores.detach().mean(dim=0).cpu().numpy()
        self._monitor_has_collab = self.has_collab

        # 5. 正交解耦惩罚 (互不干涉原则)
        loss_ortho = self.ortho_loss(h_t, h_s) + self.ortho_loss(h_i, h_s) + self.ortho_loss(h_t, h_i)
        if self.has_collab:
            loss_ortho += self.ortho_loss(h_c, h_s) + self.ortho_loss(h_t, h_c) + self.ortho_loss(h_i, h_c)

        # 6. 专家得分加权融合
        if self.has_collab:
            fused = (gate_scores[:, 0:1] * h_t) + \
                    (gate_scores[:, 1:2] * h_i) + \
                    (gate_scores[:, 2:3] * h_c) + \
                    (gate_scores[:, 3:].sum(dim=-1, keepdim=True) * h_s)
            loss_rec = F.mse_loss(self.decoder_t(fused), x_t) + \
                       F.mse_loss(self.decoder_i(fused), x_i) + \
                       F.mse_loss(self.decoder_c(fused), x_c_proj)
        else:
            fused = (gate_scores[:, 0:1] * h_t) + \
                    (gate_scores[:, 1:2] * h_i) + \
                    (gate_scores[:, 2:].sum(dim=-1, keepdim=True) * h_s)
            loss_rec = F.mse_loss(self.decoder_t(fused), x_t) + \
                       F.mse_loss(self.decoder_i(fused), x_i)

        # total_loss = loss_rec + self.Orthogonal_Loss_Weight * loss_ortho
        # 6. 动态计算总损失（Kendall 自动权衡公式： L = exp(-s)*loss + s）
        # 如果某个 loss 非常难优化(波动大)，网络会自动增加其对数方差(log_var)，从而降低该 loss 的当前权重，不至于“带偏”整个模型。
        loss_rec_dyn = 0.5 * torch.exp(-self.log_var_rec) * loss_rec + self.log_var_rec
        loss_ortho_dyn = 0.5 * torch.exp(-self.log_var_ortho) * loss_ortho + self.log_var_ortho
        
        total_loss = loss_rec_dyn + loss_ortho_dyn

        # ✨ 无损缓存当前的“等效正交权重”，供外部主程序打印观察
        # 指标换算：相当于正交 Loss 目前在总关注度中占了 重构 Loss 的多少倍
        self._monitor_ortho_weight = (torch.exp(-self.log_var_ortho) / torch.exp(-self.log_var_rec)).item()
        self._monitor_loss_rec_real = loss_rec.item()
        self._monitor_loss_ortho_real = loss_ortho.item()
        return fused, total_loss


class ContrastiveAdapter(nn.Module):
    """【无缝挪出的极简对比学习适配网络】"""
    def __init__(self, input_dim, proj_dim=None):
        super().__init__()
        if proj_dim is None:
            proj_dim = input_dim
        self.text_proj = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.image_proj = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, text_features, image_features):
        z_t = F.normalize(self.text_proj(text_features), p=2, dim=-1)
        z_i = F.normalize(self.image_proj(image_features), p=2, dim=-1)
        t = torch.clamp(self.temperature, min=1e-4) 
        logits_per_image = (z_i @ z_t.T) / t
        logits_per_text = (z_t @ z_i.T) / t
        labels = torch.arange(z_t.size(0), device=z_t.device)
        return (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2.0

    @torch.no_grad()
    def get_fused_features(self, text_features, image_features):
        return self.text_proj(text_features) + self.image_proj(image_features)