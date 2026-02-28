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
    【双曲层级蕴含损失 - 破除死锁版】
    加入了 Norm Penalty，彻底解决特征在原点附近时 violation=0 导致的无梯度死锁。
    """
    def __init__(self, aperture_thresh=0.7, norm_weight=0.1):
        super().__init__()
        # 跨模态/层级推荐阈值 (0.7 或 1.2)
        self.aperture_thresh = aperture_thresh
        # 范数惩罚的权重
        self.norm_weight = norm_weight

    def forward(self, child_emb, parent_emb, curv):
        # 1. 角度锥体约束 (Cone Penalty)
        angle = L.oxy_angle(parent_emb, child_emb, curv)
        aperture = L.half_aperture(parent_emb, curv)
        
        # 如果特征堆在原点，这里的 violation 极大概率是 0
        violation = torch.clamp(angle - self.aperture_thresh * aperture, min=0.0)
        
        # 2. 【核心修复】：原点死锁破除器 (Norm Penalty)
        # 根据双曲层级理论，子节点（更具体的概念）必须比父节点离原点更远。
        # 加入一个 0.1 的 margin，强迫子节点的范数必须大于父节点。
        # 只要它们敢停留在原点，这个 penalty 就会源源不断地提供向外的梯度！
        norm_child = L.safe_norm(child_emb, dim=-1)
        norm_parent = L.safe_norm(parent_emb, dim=-1)
        norm_penalty = torch.clamp(norm_parent - norm_child + 0.1, min=0.0)
        
        # 两者结合，彻底斩断死锁链条
        return (violation + self.norm_weight * norm_penalty).mean()


class HyperbolicHardNegativeAlignmentLoss(nn.Module):
    """
    【双曲语义级判别对齐损失】(Semantic Discriminative Alignment)
    在共享粗粒度语义 (Coarse-level) 的难负样本集中进行排斥。
    """
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, logits, labels, hard_mask=None):
        # logits 是模型输出的双曲负距离 (-dist)
        # 还原为真实的双曲黎曼距离 d(x, y)
        dists = -logits
        
        # 1. 获取正确的正样本对距离 d(v, t^+)
        pos_dists = dists.gather(1, labels.view(-1, 1))
        
        # 2. 确定候选的负样本集合
        if hard_mask is None:
            # 如果不传 mask，回退到全局难负样本
            mask = torch.ones_like(dists, dtype=torch.bool)
            mask.scatter_(1, labels.view(-1, 1), False)
        else:
            # 使用传入的语义难负样本 Mask
            mask = hard_mask
        
        # 将不允许作为负样本的位置设为无穷大，这样在取 min 时绝对不会选中它们
        neg_dists_only = dists.masked_fill(~mask, float('inf'))
        
        # 寻找“语义簇内”最难的那个负样本
        hard_neg_dists, _ = neg_dists_only.min(dim=1, keepdim=True)
        
        # 3. Triplet Margin 公式计算排斥力
        loss = torch.clamp(pos_dists - hard_neg_dists + self.margin, min=0.0)
        return loss.mean()