import pandas as pd
from src.config import Config
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from dataclasses import dataclass
from torch import Tensor

@dataclass
class DataSplit:
    X_train: Tensor
    X_test: Tensor
    y_train: Tensor
    y_test: Tensor

@dataclass
class ProcessedData:
    data: DataSplit
    scaler: StandardScaler
    encoder: OneHotEncoder

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "danceability",
        "loudness",
        "key",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
    ]
    target = Config.TARGET_COLUMN

    df_model = df[features + [target]]
    return df_model

def encode_feature(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_encoded: np.ndarray = encoder.fit_transform(X_train)
    X_test_encoded: np.ndarray = encoder.transform(X_test)
    return pd.DataFrame(
            X_train_encoded, 
            columns=encoder.get_feature_names_out(), 
            index=X_train.index
        ), pd.DataFrame(
            X_test_encoded, 
            columns=encoder.get_feature_names_out(), 
            index=X_test.index
        ), encoder

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled: np.ndarray = scaler.fit_transform(X_train)
    X_test_scaled: np.ndarray = scaler.transform(X_test)
    return pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index), pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index), scaler

def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[Config.TARGET_COLUMN])
    y = df[Config.TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    return X_train, X_test, y_train, y_test

def convert_to_tensor(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> DataSplit:
    X_train_tensor = torch.tensor(
        X_train.values,
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test.values,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train.values,
        dtype=torch.float32
    ).view(-1, 1)

    y_test_tensor = torch.tensor(
        y_test.values,
        dtype=torch.float32
    ).view(-1, 1)

    return DataSplit(
        X_train=X_train_tensor,
        X_test=X_test_tensor,
        y_train=y_train_tensor,
        y_test=y_test_tensor
    )

def process_data() -> ProcessedData:
    df = load_data(Config.DATA_PATH)
    df_model = select_features(df)
    X_train, X_test, y_train, y_test = split(df_model)
    
    X_train_encoded, X_test_encoded, encoder = encode_feature(X_train[Config.CATEGORICAL_FEATURES], X_test[Config.CATEGORICAL_FEATURES])
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train[Config.NUMERICAL_FEATURES], X_test[Config.NUMERICAL_FEATURES])

    X_train_processed = pd.concat([X_train_scaled, X_train_encoded], axis=1)
    X_test_processed = pd.concat([X_test_scaled, X_test_encoded], axis=1)

    return ProcessedData(
        data = convert_to_tensor(X_train_processed, X_test_processed, y_train, y_test),
        scaler=scaler,
        encoder=encoder,
    )
    
