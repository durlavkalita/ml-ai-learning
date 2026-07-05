import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from dataclasses import dataclass
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)

@dataclass
class EvaluationResult:
    mse: float
    mae: float
    r2: float
    
def predict(
    model: Module,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:

    model.to(device)
    model.eval()

    predictions: list[np.ndarray] = []
    actuals: list[np.ndarray] = []

    with torch.no_grad():

        for X_batch, y_batch in dataloader:

            X_batch = X_batch.to(device)

            outputs = model(X_batch)

            predictions.append(outputs.cpu().numpy())

            actuals.append(y_batch.numpy())

    return (
        np.concatenate(predictions),
        np.concatenate(actuals),
    )

def calculate_r2(
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> float:
    return r2_score(actuals, predictions)

def evaluate_model(
    model: Module,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
) -> EvaluationResult:

    predictions, actuals = predict(
        model,
        dataloader,
        device,
    )

    mse = mean_squared_error(
        actuals,
        predictions,
    )

    mae = mean_absolute_error(
        actuals,
        predictions,
    )

    r2 = calculate_r2(
        predictions,
        actuals,
    )

    return EvaluationResult(
        mse=mse,
        mae=mae,
        r2=r2,
    )