import torch
import torch.nn as nn

# --- 引入双曲底层算子库 ---
import models.vlm_models.lorentz as L
# -----------------------------------------------------------------------

# =============================================================================
# [PURE HYPERBOLIC] 核心双曲损失组件
# =============================================================================

class EntailmentConeLoss(nn.Module):
    """
    【双曲层级蕴含损失 - 纯净版】
    依赖 L.half_aperture 内部的 min_radius 解决死锁，彻底移除了暴力的范数惩罚。
    让模型优先专注于语义聚类，而不是被强制推离原点。
    """
    def __init__(self, aperture_thresh=0.7):
        super().__init__()
        # 跨模态/层级推荐阈值 (0.7 或 1.2)
        self.aperture_thresh = aperture_thresh

    def forward(self, child_emb, parent_emb, curv):
        # 计算子节点在父节点视角的夹角和父节点的孔径
        angle = L.oxy_angle(child_emb, parent_emb, curv)
        aperture = L.half_aperture(parent_emb, curv)
        
        # 只要夹角超出了阈值孔径，就产生损失。移除惩罚范数的额外拉扯。
        violation = torch.clamp(angle - self.aperture_thresh * aperture, min=0.0)
        
        return violation.mean()


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