import torch
from torch.utils.data import Dataset

class NextTokenDataset(Dataset):
    """
    PyTorch Dataset для задачи предсказания следующего токена
    """
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.X[idx],
            'labels': self.y[idx]
        }
