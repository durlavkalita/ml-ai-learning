from pathlib import Path
from typing import Any

import joblib
import torch
from torch.nn import Module

def save_model(
    model: Module,
    path: str | Path
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        path
    )

def load_model(
    model: Module,
    path: str | Path,
    device: torch.device
)-> Module:
    state_dict = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def save_object(
    obj: Any,
    path: str | Path,
) -> None:

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        obj,
        path,
    )

def load_object(
    path: str | Path,
) -> Any:

    return joblib.load(path)