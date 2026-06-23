import torch
import torch.nn.functional as F

def Norm(x):
    """将特征图归一化到 0~255 的浮点数 (保留梯度)"""
    B, C, H, W = x.shape
    x_flat = x.view(B, C, -1)
    min_val = x_flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
    max_val = x_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    delta = (max_val - min_val).clamp(min=1e-8)
    return (x - min_val) / delta * 255.0

def Entropy(x):
    """极速版信息熵计算"""
    B, C, H, W = x.shape
    x_int = x.round().long().clamp(min=0, max=255)
    # 用 one_hot 替代 256 次 for 循环，速度起飞
    histic = F.one_hot(x_int, num_classes=256).sum(dim=(2, 3)).float()
    p_ij = histic / (H * W)
    h_ij = -torch.sum(p_ij * torch.log(p_ij + 1e-8), dim=2)  
    return h_ij.unsqueeze(-1).unsqueeze(-1)

def Cos_Similarity(x, y):
    """修复版空间余弦相似度"""
    B, C, H, W = x.shape
    cos = F.cosine_similarity(x.view(B, C, -1), y.view(B, C, -1), dim=2) 
    return cos.unsqueeze(-1).unsqueeze(-1)

def Smooth(tensor, a=0.02):
    """平滑L1规范化"""
    temp_norm = torch.norm(tensor.float(), p=2, dim=1) 
    map_1, map_s = (temp_norm >= a).float(), (temp_norm < a).float()
    return map_1 * temp_norm + map_s * (0.5 * torch.pow(temp_norm, 2) / a + a * 0.5)

def Complementary_Learning_Loss(ms_v, pan_v):
    ms_v, pan_v = torch.mean(ms_v, dim=1, keepdim=True), torch.mean(pan_v, dim=1, keepdim=True)
    
    cos_similarity = Cos_Similarity(ms_v, pan_v)
    x_diff = Entropy(Norm(ms_v)) - Entropy(Norm(pan_v))
    k = 1.0 / (torch.pow(cos_similarity, 2) + 0.001)

    w1 = torch.sigmoid(k * x_diff)
    w2 = 1.0 - w1

    loss_map = w1.squeeze(1) * Smooth(ms_v) + w2.squeeze(1) * Smooth(pan_v)
    return loss_map.mean() / 10.0 # 强制返回标量