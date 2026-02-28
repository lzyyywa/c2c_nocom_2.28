import math
import torch
from torch import Tensor
from typing import Union  # 引入兼容老版本的 Union

def pairwise_inner(x: Tensor, y: Tensor, curv: Union[float, Tensor] = 1.0):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
    xyl = x @ y.T - x_time @ y_time.T
    return xyl

def pairwise_dist(x: Tensor, y: Tensor, curv: Union[float, Tensor] = 1.0, eps: float = 1e-8) -> Tensor:
    c_xyl = -curv * pairwise_inner(x, y, curv)
    # 【必须的保护】acosh 的输入必须 >= 1
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.5

def exp_map0(x: Tensor, curv: Union[float, Tensor] = 1.0, eps: float = 1e-8) -> Tensor:
    # 【Norm-based 截断】绝不能单独截断坐标维度，只能限制范数
    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15)) # 防止 sinh 爆炸
    _output = torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)
    return _output

def half_aperture(x: Tensor, curv: Union[float, Tensor] = 1.0, min_radius: float = 0.1, eps: float = 1e-8) -> Tensor:
    # 【原点死锁保护】假定原点有一个 min_radius，特征再小也能算出孔径
    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv**0.5 + eps)
    _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))
    return _half_aperture

def oxy_angle(x: Tensor, y: Tensor, curv: Union[float, Tensor] = 1.0, eps: float = 1e-8):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))
    return _angle