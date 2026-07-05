import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch import Tensor
from dataclasses import dataclass
import copy
from src.config import Config

@dataclass
class TrainResult:
	model: nn.Module
	train_losses: list[float]
	val_losses: list[float]

def train_and_validate_model(
		model: nn.Module, 
		train_loader: DataLoader[tuple[Tensor,Tensor]], 
		val_loader: DataLoader[tuple[Tensor, Tensor]], 
		epochs: int = Config.EPOCHS, 
		lr: float = Config.LEARNING_RATE, 
		device: torch.device|None = None, 
		criterion: nn.Module | None = None, 
		optimizer: torch.optim.Optimizer | None = None
	) -> TrainResult:

	if device is None:
		device = torch.device(
			"cuda" if torch.cuda.is_available()
			else "cpu"
		)
	model.to(device)

	if criterion is None:
		criterion = nn.MSELoss()

	if optimizer is None:
		optimizer = torch.optim.Adam(
			model.parameters(),
			lr = lr
		)
	
	train_losses: list[float] = []
	val_losses: list[float] = []
	best_model_state = copy.deepcopy(model.state_dict())
	best_val_loss = float("inf")	
	best_epoch = 0

	for epoch in range(epochs):
		model.train()
		epoch_loss = 0
		for X_batch, y_batch in train_loader:

			X_batch = X_batch.to(device)
			y_batch = y_batch.to(device)

			predictions = model(X_batch)

			loss = criterion(
				predictions,
				y_batch
			)

			optimizer.zero_grad()

			loss.backward()

			optimizer.step()
			
			epoch_loss += loss.item()

		avg_loss = epoch_loss / len(train_loader)
		train_losses.append(avg_loss)

		model.eval()
		val_loss = 0
		with torch.no_grad():
			for X_batch, y_batch in val_loader:
				X_batch = X_batch.to(device)
				y_batch = y_batch.to(device)

				predictions = model(X_batch)
				loss = criterion(
					predictions,
					y_batch
				)
				val_loss += loss.item()

		avg_val_loss = val_loss / len(val_loader)
		val_losses.append(avg_val_loss)

		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			best_epoch = epoch
			best_model_state = copy.deepcopy(model.state_dict())

		if epoch%5 == 0:
			print(f"Epoch: {epoch:2d}/{epochs} ; Train Loss: {avg_loss:.5f} ; Val Loss: {avg_val_loss:.5f}")
	model.load_state_dict(best_model_state)
	print(
		f"Loaded best model from epoch {best_epoch} "
		f"(val_loss={best_val_loss:.5f})"
	)
	return TrainResult(model, train_losses, val_losses)