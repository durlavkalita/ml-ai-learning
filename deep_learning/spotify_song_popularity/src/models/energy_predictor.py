import torch
import torch.nn as nn
from typing import List

class EnergyPredictor(nn.Module):
    
	def __init__(self, input_size:int = 51, hidden_size:int = 2, use_relu:bool = False):
		super().__init__()

		layers: List[nn.Linear | nn.ReLU] = [
			nn.Linear(input_size, hidden_size)
		]

		if use_relu:
			layers.append(nn.ReLU())
		
		layers.append(
			nn.Linear(hidden_size, 1)
		)

		self.network = nn.Sequential(*layers)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.network(x)
