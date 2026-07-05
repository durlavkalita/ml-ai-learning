import torch.nn as nn
import torch 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from torch import Tensor
import numpy as np
from src.utils.types import SongFeatures


class EnergyPredictorService:

    def __init__(
        self,
        model: nn.Module,
        scaler: StandardScaler,
        encoder: OneHotEncoder,
        device: torch.device,
    ):
        self.model = model
        self.scaler = scaler
        self.encoder = encoder
        self.device = device

    def _preprocess(
        self,
        song: SongFeatures,
    ) -> Tensor:
        numerical = pd.DataFrame([
            {
                "danceability": 0.81,
                "loudness": -4.5,
                "speechiness": 0.3150,
                "acousticness": 0.00740,
                "instrumentalness": 0.838000,
                "liveness": 0.4790,
                "valence": 0.000,
                "tempo": 123.588,
                "duration_ms": 79500.0
            }
        ])
        scaled = self.scaler.transform(numerical)
        categorical = pd.DataFrame([
            {
                "key": song.key,
                "mode": song.mode,
            }
        ])
        encoded = self.encoder.transform(categorical)
        processed = np.hstack([
            scaled,
            encoded,
        ])
        return torch.tensor(
            processed,
            dtype=torch.float32,
            device=self.device,
        )
    
    def predict(
    self,
    song: SongFeatures,
    ) -> float:

        tensor = self._preprocess(song)

        self.model.eval()

        with torch.no_grad():

            prediction = self.model(tensor)

        return prediction.item()
