import torch
import torch.nn as nn
import torch.optim as optim

# Definition of Logistic Regression
class AdaptiveClassifier(nn.Module):
  def __init__(self, input_dim = 1):
    super().__init__()
    self.linear = nn.Linear(input_dim, 1)

  def forward(self, x):
    return torch.sigmoid(self.linear(x))