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
    【完全对齐 HyCoCLIP】双曲层级蕴含损失 (Aperture-Thresholded Entailment)
    放弃人为构造的 norm_penalty，采用官方的“孔径缩水”策略强迫特征远离原点。
    
    原理：在双曲空间中，越靠近原点，能覆盖的孔径（aperture）越大（接近180度）。
    如果我们人为将孔径缩小（例如乘以 0.7），而在原点附近的特征因为重叠导致夹角变大，
    就会直接越过这个变窄的孔径产生巨大的 Loss。
    为了降低 Loss，模型唯一的出路就是将父子节点整体向双曲空间的边缘推。
    """
    def __init__(self, aperture_thresh=0.7):
        super().__init__()
        # HyCoCLIP 官方跨模态/层级推荐阈值 (例如 0.7 或 1.2)
        self.aperture_thresh = aperture_thresh

    def forward(self, child_emb, parent_emb, curv):
        # 1. 计算父节点到子节点的实际夹角
        angle = L.oxy_angle(parent_emb, child_emb, curv)
        
        # 2. 计算父节点的理论半顶角 (孔径)
        aperture = L.half_aperture(parent_emb, curv)
        
        # 3. HyCoCLIP 核心防坍缩逻辑：孔径缩水
        # 要求实际夹角必须小于缩水后的孔径。
        violation = torch.clamp(angle - self.aperture_thresh * aperture, min=0.0)
        
        return violation.mean()


class HyperbolicHardNegativeAlignmentLoss(nn.Module):
    """
    【双曲语义级判别对齐损失】(Semantic Discriminative Alignment)
    完全对齐 H2EM/HyCoCLIP 思想：在共享粗粒度语义 (Coarse-level) 的难负样本集中进行排斥。
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
            # 如果不传 mask，回退到全局难负样本 (Global Hard Negative)
            mask = torch.ones_like(dists, dtype=torch.bool)
            mask.scatter_(1, labels.view(-1, 1), False)
        else:
            # [核心逻辑]：使用传入的语义难负样本 Mask！(只考虑同粗类下的其他细类)
            mask = hard_mask
        
        # 将不允许作为负样本的位置设为无穷大，这样在取 min 时绝对不会选中它们
        neg_dists_only = dists.masked_fill(~mask, float('inf'))
        
        # 寻找“语义簇内”最难的那个负样本
        hard_neg_dists, _ = neg_dists_only.min(dim=1, keepdim=True)
        
        # 3. Triplet Margin 公式计算排斥力
        loss = torch.clamp(pos_dists - hard_neg_dists + self.margin, min=0.0)
        return loss.mean()