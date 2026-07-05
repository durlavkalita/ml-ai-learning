from torch.utils.data import Dataset
from torch import Tensor
from typing import Tuple

class SpotifyDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, X: Tensor, y: Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.X[index], self.y[index]
    
