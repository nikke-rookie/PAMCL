import torch
import numpy as np
class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X, device: torch.device):
        """
        将稀疏矩阵转换为PyTorch稀疏张量

        Args:
            X (scipy.sparse.coo_matrix): 稀疏矩阵
            device (torch.device): GPU
        
        Returns:
            torch.sparse_coo_tensor
        """
        coo = X.tocoo()
        coords = np.array([coo.row, coo.col])
        i = torch.tensor(coords, dtype=torch.long, device=device)
        v = torch.from_numpy(coo.data).float().to(device)
        return torch.sparse_coo_tensor(i, v, coo.shape, device=device)
