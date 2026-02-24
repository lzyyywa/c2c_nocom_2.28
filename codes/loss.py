from torch.nn.modules.loss import CrossEntropyLoss

# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

# --- 引入双曲底层算子库 ---
import models.vlm_models.lorentz as L
# -----------------------------------------------------------------------

def loss_calu(predict, target, config):
    loss_fn = CrossEntropyLoss()
    batch_img, batch_attr, batch_obj, batch_target = target
    batch_attr = batch_attr.cuda()
    batch_obj = batch_obj.cuda()
    batch_target = batch_target.cuda()
    logits, logits_att, logits_obj, logits_soft_prompt = predict
    loss_logit_df = loss_fn(logits, batch_target)
    loss_logit_sp = loss_fn(logits_soft_prompt, batch_target)
    loss_att = loss_fn(logits_att, batch_attr)
    loss_obj = loss_fn(logits_obj, batch_obj)
    
    att_obj_w = config.att_obj_w if hasattr(config, 'att_obj_w') else 0.2
    sp_w = getattr(config, 'sp_w', 0.0)
    loss = loss_logit_df + att_obj_w * (loss_att + loss_obj) + sp_w * loss_logit_sp
    return loss

# =============================================================================
# [HYPERBOLIC] 核心双曲损失组件 (严格对齐 Troika & HyCoCLIP)
# =============================================================================

class EntailmentConeLoss(nn.Module):
    """
    【双曲层级蕴含损失】(Hierarchical Entailment Cone Loss)
    强迫细粒度子概念进入粗粒度父概念的蕴含锥 (Cone) 内。
    包含：1. 角度惩罚 (Cone Penalty)  2. 范数惩罚 (Norm Penalty)
    """
    def __init__(self, margin=0.01):
        super().__init__()
        self.margin = margin

    def forward(self, child_emb, parent_emb, curv):
        # 1. 计算父节点的半顶角与实际夹角
        angle = L.oxy_angle(parent_emb, child_emb, curv)
        aperture = L.half_aperture(parent_emb, curv)
        
        # 若夹角大于半顶角(越界)，则产生 violation 惩罚
        violation = torch.clamp(angle - aperture + self.margin, min=0.0)
        
        # 2. 范数惩罚: 强迫子节点(更具体)距离原点更远
        norm_child = torch.norm(child_emb, dim=-1)
        norm_parent = torch.norm(parent_emb, dim=-1)
        norm_penalty = torch.clamp(norm_parent - norm_child, min=0.0)
        
        return (violation + 0.1 * norm_penalty).mean()


class HyperbolicHardNegativeAlignmentLoss(nn.Module):
    """
    【双曲判别对齐损失】(Discriminative Alignment with Hard Negative Mining)
    彻底修复了特征坍缩问题！
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


def hyperbolic_loss_calu(predict, target):
    """
    仅负责计算数学公式，算出 5 个纯净的损失分量，绝不进行任何系数加权！
    """
    loss_fn = CrossEntropyLoss()
    entail_loss_fn = EntailmentConeLoss(margin=0.01)
    align_loss_fn = HyperbolicHardNegativeAlignmentLoss(margin=0.2) 

    # 1. 解包数据
    logits_v, logits_o, logits_com, hyp_feats = predict
    batch_img, batch_attr, batch_obj, batch_target, coarse_v_hyp, coarse_o_hyp = target
    
    batch_attr = batch_attr.cuda()
    batch_obj = batch_obj.cuda()
    batch_target = batch_target.cuda()

    # =========================================================================
    # Part 1: 基础分类损失 (Base CE Loss) - [com, v, o]
    # 使用温度缩放 20.0 对齐基线的分类能力
    # =========================================================================
    loss_com_ce = loss_fn(logits_com.float() * 20.0, batch_target)
    loss_v_ce = loss_fn(logits_v.float() * 20.0, batch_attr)
    loss_o_ce = loss_fn(logits_o.float() * 20.0, batch_obj)
    
    # =========================================================================
    # Part 2: 难负样本判别对齐 (Discriminative Alignment) - [com, v, o]
    # =========================================================================
    loss_align_v = align_loss_fn(logits_v.float(), batch_attr)
    loss_align_o = align_loss_fn(logits_o.float(), batch_obj)
    loss_align_com = align_loss_fn(logits_com.float(), batch_target)
    
    loss_align_pack = (loss_align_com, loss_align_v, loss_align_o)
    
    # =========================================================================
    # Part 3: 层次蕴含 (Hierarchical Entailment) - [细->粗, 动作->细]
    # =========================================================================
    curv = hyp_feats['curv']
    
    # [关键] 索引出当前 batch 对应的细粒度文本特征作为锥体中间层
    batch_fine_v_hyp = hyp_feats['verb_text_hyp'][batch_attr]
    batch_fine_o_hyp = hyp_feats['obj_text_hyp'][batch_obj]
    
    # [Level 1]: 视频特征(child) -> 细粒度文本(parent)
    loss_entail_v_1 = entail_loss_fn(hyp_feats['v_feat_hyp'], batch_fine_v_hyp, curv)
    loss_entail_o_1 = entail_loss_fn(hyp_feats['o_feat_hyp'], batch_fine_o_hyp, curv)
    
    # [Level 2]: 细粒度文本(child) -> 粗粒度文本(parent)
    loss_entail_v_2 = entail_loss_fn(batch_fine_v_hyp, coarse_v_hyp, curv)
    loss_entail_o_2 = entail_loss_fn(batch_fine_o_hyp, coarse_o_hyp, curv)
    
    # 蕴含损失直接相加
    loss_entail_total = (loss_entail_v_1 + loss_entail_o_1 + loss_entail_v_2 + loss_entail_o_2)

    # 打包返回所有的基础标量，交还给 train_models.py 进行显式加权
    return loss_com_ce, loss_v_ce, loss_o_ce, loss_align_pack, loss_entail_total


# =============================================================================
# 原有备用函数保持原样 (KLLoss, hsic_loss, Gml_loss)
# =============================================================================

class KLLoss(nn.Module):
    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        self.error_metric = error_metric

    def forward(self, prediction, label,mul=False):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label, 1)
        loss = self.error_metric(probs1, probs2)
        if mul:
            return loss* batch_size
        else:
            return loss


def hsic_loss(input1, input2, unbiased=False):
    def _kernel(X, sigma):
        X = X.view(len(X), -1)
        XX = X @ X.t()
        X_sqnorms = torch.diag(XX)
        X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        gamma = 1 / (2 * sigma ** 2)

        kernel_XX = torch.exp(-gamma * X_L2)
        return kernel_XX

    N = len(input1)
    if N < 4:
        return torch.tensor(0.0).to(input1.device)
    sigma_x = np.sqrt(input1.size()[1])
    sigma_y = np.sqrt(input2.size()[1])

    kernel_XX = _kernel(input1, sigma_x)
    kernel_YY = _kernel(input2, sigma_y)

    if unbiased:
        tK = kernel_XX - torch.diag(kernel_XX)
        tL = kernel_YY - torch.diag(kernel_YY)
        hsic = (
                torch.trace(tK @ tL)
                + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
                - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
        )
        loss = hsic / (N * (N - 3))
    else:
        KH = kernel_XX - kernel_XX.mean(0, keepdim=True)
        LH = kernel_YY - kernel_YY.mean(0, keepdim=True)
        loss = torch.trace(KH @ LH / (N - 1) ** 2)
    return loss


class Gml_loss(nn.Module):
    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()

    def forward(self, p_o_on_v, v_label, n_c, t=100.0):
        n_c = n_c[:, 0]
        b = p_o_on_v.shape[0]
        n_o = p_o_on_v.shape[-1]
        p_o = p_o_on_v[range(b), v_label, :]  

        num_c = n_c.sum().view(1, -1)  

        p_o_exp = torch.exp(p_o * t)
        p_o_exp_wed = num_c * p_o_exp  
        p_phi = p_o_exp_wed / torch.sum(p_o_exp_wed, dim=0, keepdim=True)  

        p_ba = torch.sum(p_phi * n_c, dim=0, keepdim=True) / (num_c + 1.0e-6)  
        p_ba[torch.where(p_ba < 1.0e-8)] = 1.0
        p_ba_log = torch.log(p_ba)
        loss = (-1.0 / n_o) * p_ba_log.sum()

        return loss