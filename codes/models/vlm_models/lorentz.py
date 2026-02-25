import math
import torch
from torch import Tensor
from typing import Union

# ======== [核心保命算子] 彻底斩断原点塌陷时的 NaN 梯度 ========
def safe_norm(x: Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-7) -> Tensor:
    return torch.sqrt(torch.clamp(torch.sum(x**2, dim=dim, keepdim=keepdim), min=eps))
# ===============================================================

def pairwise_inner(x: Tensor, y: Tensor, curv: Union[float, Tensor] = 1.0):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
    xyl = x @ y.T - x_time @ y_time.T
    return torch.nan_to_num(xyl)

def pairwise_dist(x: Tensor, y: Tensor, curv: Union[float, Tensor] = 1.0, eps: float = 1e-5) -> Tensor:
    c_xyl = -curv * pairwise_inner(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return torch.nan_to_num(_distance / curv**0.5)

def exp_map0(x: Tensor, curv: Union[float, Tensor] = 1.0, eps: float = 1e-5) -> Tensor:
    rc_xnorm = curv**0.5 * safe_norm(x, dim=-1, keepdim=True)
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
    _output = torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)
    return torch.nan_to_num(_output)

def log_map0(x: Tensor, curv: Union[float, Tensor] = 1.0, eps: float = 1e-5) -> Tensor:
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
    _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))
    rc_xnorm = curv**0.5 * safe_norm(x, dim=-1, keepdim=True)
    _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
    return torch.nan_to_num(_output)

def half_aperture(x: Tensor, curv: Union[float, Tensor] = 1.0, min_radius: float = 0.1, eps: float = 1e-5) -> Tensor:
    asin_input = 2 * min_radius / (safe_norm(x, dim=-1) * curv**0.5 + eps)
    _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))
    return torch.nan_to_num(_half_aperture)

def oxy_angle(x: Tensor, y: Tensor, curv: Union[float, Tensor] = 1.0, eps: float = 1e-5):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

    acos_input = acos_numer / (safe_norm(x, dim=-1) * acos_denom + eps)
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))
    return torch.nan_to_num(_angle)

def oxy_angle_eval(x: Tensor, y: Tensor, curv: Union[float, Tensor] = 1.0, eps: float = 1e-5):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))

    c_xyl = curv * (y @ x.T - y_time @ x_time.T)

    acos_numer = y_time + c_xyl * x_time.T
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

    acos_input = acos_numer / (safe_norm(x, dim=-1, keepdim=True).T * acos_denom + eps)
    _angle = - torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))
    return torch.nan_to_num(_angle)