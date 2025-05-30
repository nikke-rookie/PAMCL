import torch
import torch.nn.functional as F
import torch.nn as nn

def bpr_loss_w(user_emb: torch.Tensor, pos_item_emb: torch.Tensor, neg_item_embs: torch.Tensor):
    """
    Bayesian Personalized Ranking (BPR) 损失函数
    fix:
        1. 多负样本(中心性权重控制)
        2. 文本模态引导负样本筛选
    
    Args:
        user_emb: 用户嵌入向量 (batch_size, embedding_dim)
        pos_item_emb: 正样本物品嵌入向量 (batch_size, embedding_dim)
        neg_item_embs: 负样本物品嵌入向量 (batch_size, n_negs, embedding_dim)
        neg_weights: 负样本权重 (batch_size, n_negs)
    
    Returns:
        BPR损失的平均值
    """
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_scores = torch.mul(user_emb.unsqueeze(1), neg_item_embs).sum(dim=2)
    loss = -torch.log(10e-6 + torch.sigmoid(pos_score.unsqueeze(1) - neg_scores))
    return torch.mean(loss)

def l2_reg_loss(reg: float, embeddings: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    """
    计算L2正则化损失

    Args:
        reg (float): 正则化系数
        embeddings (List[torch.Tensor]): 需要进行正则化的张量列表
    
    Returns:
        正则化损失的平均值，标量张量
    """
    emb_loss = torch.tensor(0., device=device)
    for emb in embeddings:
        emb_loss += torch.linalg.vector_norm(emb, ord=2) / emb.shape[0]
        emb_loss += 1./2*torch.sum(emb**2) / emb.shape[0]
    return emb_loss * reg


def InfoNCE(view1: torch.Tensor, view2: torch.Tensor, temperature: float, b_cos: bool = True):
    """
    计算InfoNCE损失函数

    Args:
        view1 (torch.Tensor): Num x Dim
        view2 (torch.Tensor): Num x Dim
        temperature (float): 温度系数
        b_cos (bool): 是否使用余弦相似度

    Returns:
        Average InfoNCE Loss
    """
    if view1.shape != view2.shape:
        raise ValueError(f"InfoNCE expected the same shape for two views. But got view1.shape={view1.shape} and view2.shape={view2.shape}.")
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 @ view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()


def cl_loss(idx: list[int], view1, view2, temp: float, device: torch.device) -> torch.Tensor:
    """
    对比学习损失函数
    
    Args:
        idx (list): [user_idx_list, pos_idx_list]
        view1: 对比视图1
        view2: 对比视图2
    
    Returns:
        cl_loss
    """
    idx_tensor = torch.unique(torch.tensor(idx, dtype=torch.long, device=device))
    loss = InfoNCE(view1[idx_tensor], view2[idx_tensor], temp)
    return loss

def cross_cl_loss(idx, view1, view2, view3, temp, device):
    idx_tensor = torch.unique(torch.tensor(idx, dtype=torch.long, device=device))
    loss1 = InfoNCE(view1[idx_tensor], view2[idx_tensor], temp)
    loss2 = InfoNCE(view1[idx_tensor], view3[idx_tensor], temp)
    return loss1 + loss2

