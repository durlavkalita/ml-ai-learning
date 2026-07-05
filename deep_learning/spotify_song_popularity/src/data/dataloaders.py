from torch.utils.data import DataLoader, Subset
from torch import Tensor
from src.data.dataset import SpotifyDataset

def create_dataloaders(X_train_tensor: Tensor, X_test_tensor: Tensor, y_train_tensor: Tensor, y_test_tensor: Tensor, batch_size: int, subset_size: int):

    train_dataset = SpotifyDataset(
        X_train_tensor,
        y_train_tensor
    )
    small_dataset = Subset(
        train_dataset,
        range(subset_size)
    )
    train_loader = DataLoader(
        small_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataset = SpotifyDataset(
        X_test_tensor,
        y_test_tensor
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
    )

    return train_loader, test_loader