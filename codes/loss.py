from torch.nn.modules.loss import CrossEntropyLoss

# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

# --- Added: Import lorentz operations for hyperbolic loss calculations ---
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
    loss = loss_logit_df + config.att_obj_w * (loss_att + loss_obj) + config.sp_w * loss_logit_sp
    return loss


class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
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
    # we simply use the squared dimension of feature as the sigma for RBF kernel
    sigma_x = np.sqrt(input1.size()[1])
    sigma_y = np.sqrt(input2.size()[1])

    # compute the kernels
    kernel_XX = _kernel(input1, sigma_x)
    kernel_YY = _kernel(input2, sigma_y)

    if unbiased:
        """Unbiased estimator of Hilbert-Schmidt Independence Criterion
        Song, Le, et al. "Feature selection via dependence maximization." 2012.
        """
        # tK = kernel_XX - torch.diag(torch.diag(kernel_XX))
        # tL = kernel_YY - torch.diag(torch.diag(kernel_YY))
        tK = kernel_XX - torch.diag(kernel_XX)
        tL = kernel_YY - torch.diag(kernel_YY)
        hsic = (
                torch.trace(tK @ tL)
                + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
                - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
        )
        loss = hsic / (N * (N - 3))
    else:
        """Biased estimator of Hilbert-Schmidt Independence Criterion
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        """
        KH = kernel_XX - kernel_XX.mean(0, keepdim=True)
        LH = kernel_YY - kernel_YY.mean(0, keepdim=True)
        loss = torch.trace(KH @ LH / (N - 1) ** 2)
    return loss


class Gml_loss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    Loss from No One Left Behind: Improving the Worst Categories in Long-Tailed Learning
    """

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()

    def forward(self, p_o_on_v, v_label, n_c, t=100.0):
        '''

        Args:
            p_o_on_v: b,n_v,n_o
            o_label: b,
            n_c: b,n_o

        Returns:

        '''
        n_c = n_c[:, 0]
        b = p_o_on_v.shape[0]
        n_o = p_o_on_v.shape[-1]
        p_o = p_o_on_v[range(b), v_label, :]  # b,n_o

        num_c = n_c.sum().view(1, -1)  # 1,n_o

        p_o_exp = torch.exp(p_o * t)
        p_o_exp_wed = num_c * p_o_exp  # b,n_o
        p_phi = p_o_exp_wed / torch.sum(p_o_exp_wed, dim=0, keepdim=True)  # b,n_o

        p_ba = torch.sum(p_phi * n_c, dim=0, keepdim=True) / (num_c + 1.0e-6)  # 1,n_o
        p_ba[torch.where(p_ba < 1.0e-8)] = 1.0
        p_ba_log = torch.log(p_ba)
        loss = (-1.0 / n_o) * p_ba_log.sum()

        return loss

# ====== Added Hyperbolic Hierarchical and Alignment Losses ======

def HierarchicalEntailmentLoss(child, parent, curv):
    """
    Computes the compositional entailment loss in hyperbolic space based on Cone Entailment.
    Enforces that the child node lies within the entailment cone of the parent node,
    and that the child has a larger hyperbolic norm (is further from the origin) than the parent.
    """
    # Calculate half-aperture angle of the parent's entailment cone
    theta_parent = L.half_aperture(parent, curv)
    
    # Calculate the exterior angle at origin between child and parent
    angle_cp = L.oxy_angle(child, parent, curv)
    
    # Cone constraint: the angle between child and parent should be smaller than half-aperture
    cone_penalty = torch.clamp(angle_cp - theta_parent, min=0.0)
    
    # Norm constraint: child should be more specific (further from origin) than parent
    norm_child = torch.norm(child, dim=-1)
    norm_parent = torch.norm(parent, dim=-1)
    norm_penalty = torch.clamp(norm_parent - norm_child, min=0.0)
    
    # Combine penalties
    return (cone_penalty + 0.1 * norm_penalty).mean()


def HyperbolicAlignmentLoss(visual_feat, text_feat, curv):
    """
    Aligns visual features and corresponding text features in hyperbolic space 
    by minimizing their pairwise distance.
    """
    # Calculate pairwise distance
    dist = L.pairwise_dist(visual_feat, text_feat, curv)
    
    # Since visual_feat and text_feat are batch-aligned, we minimize the diagonal (positive pairs)
    return dist.diag().mean()

# ================================================================