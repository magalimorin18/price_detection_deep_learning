"""Model Class CNN."""
import torch
import torch.nn.functional as F
from torch import nn

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNNet(nn.Module):
    """Class CNN."""

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 10, 5, 1)
        self.conv_2 = nn.Conv2d(10, 20, 5, 1)
        self.fc_1 = nn.Linear(4 * 4 * 20, 500)
        self.fc_2 = nn.Linear(500, 10)

    def forward(self, x):
        """Fonction forward"""
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 20)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x
