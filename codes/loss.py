import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss

# --- 引入双曲底层算子库 ---
import models.vlm_models.lorentz as L
# -----------------------------------------------------------------------

# =============================================================================
# [PURE HYPERBOLIC] 核心双曲损失组件
# =============================================================================

class EntailmentConeLoss(nn.Module):
    """
    【双曲层级蕴含损失】(Hierarchical Entailment Cone Loss)
    强迫细粒度子概念进入粗粒度父概念的蕴含锥 (Cone) 内。
    包含：1. 角度惩罚 (Cone Penalty)  2. 范数惩罚 (Norm Penalty)
    [已修复]: 引入 safe_norm，防止特征趋近原点时梯度为 NaN
    """
    def __init__(self, margin=0.01):
        super().__init__()
        self.margin = margin

    # 本地化防崩塌算子
    def safe_norm(self, x, dim=-1):
        # 强制加入 eps=1e-7，斩断 sqrt(0) 求导时分母为 0 的可能性
        return torch.sqrt(torch.clamp(torch.sum(x**2, dim=dim), min=1e-7))

    def forward(self, child_emb, parent_emb, curv):
        # 1. 计算父节点的半顶角与实际夹角
        angle = L.oxy_angle(parent_emb, child_emb, curv)
        aperture = L.half_aperture(parent_emb, curv)
        
        # 若夹角大于半顶角(越界)，则产生 violation 惩罚
        violation = torch.clamp(angle - aperture + self.margin, min=0.0)
        
        # 2. 范数惩罚: 强迫子节点(更具体)距离原点更远
        # [严防死守] 替换原有的 torch.norm 为 safe_norm
        norm_child = self.safe_norm(child_emb, dim=-1)
        norm_parent = self.safe_norm(parent_emb, dim=-1)
        norm_penalty = torch.clamp(norm_parent - norm_child, min=0.0)
        
        return (violation + 0.1 * norm_penalty).mean()


class HyperbolicHardNegativeAlignmentLoss(nn.Module):
    """
    【双曲判别对齐损失】(Discriminative Alignment with Hard Negative Mining)
    严格按照论文公式: L = max(0, d(v, t^+) - d(v, t^-_{hard}) + margin)
    """
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, logits, labels):
        # logits 是模型输出的双曲负距离 (-dist)
        # 还原为真实的双曲黎曼距离 d(x, y)
        dists = -logits
        
        # 1. 获取正确的正样本对距离 d(v, t^+)
        pos_dists = dists.gather(1, labels.view(-1, 1))
        
        # 2. 寻找最难负样本对距离 d(v, t^-_{hard}) 
        # (即除了正确标签外，距离当前视频最近的那个错误类别)
        mask = torch.ones_like(dists, dtype=torch.bool)
        mask.scatter_(1, labels.view(-1, 1), False) # 把正确标签位置屏蔽掉
        
        # 将正样本距离设为无穷大，从而在取 min 时绝对只选出负样本
        neg_dists_only = dists.masked_fill(~mask, float('inf'))
        hard_neg_dists, _ = neg_dists_only.min(dim=1, keepdim=True)
        
        # 3. Triplet Margin 公式计算排斥力
        loss = torch.clamp(pos_dists - hard_neg_dists + self.margin, min=0.0)
        return loss.mean()