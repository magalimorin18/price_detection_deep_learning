"""Model Class CNN."""
import torch
import torch.nn.functional as F
from torch import nn

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")
# +
DATA_SIZE = 784
NUM_HIDDEN_1 = 256  # try 512
NUM_HIDDEN_2 = 256
NUM_CLASSES = 10

NUM_CONV_1 = 10  # try 32
NUM_CONV_2 = 20  # try 64
NUM_FC = 500  # try 1024


class CNNNet(nn.Module):
    """Class CNN."""

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, NUM_CONV_1, 5, 1)  # kernel_size = 5
        self.conv_2 = nn.Conv2d(NUM_CONV_1, NUM_CONV_2, 5, 1)  # kernel_size = 5
        # self.drop = nn.Dropout2d()
        self.fc_1 = nn.Linear(4 * 4 * NUM_CONV_2, NUM_FC)
        self.fc_2 = nn.Linear(NUM_FC, NUM_CLASSES)

    def forward(self, x):
        """Fonction forward"""
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        # x = F.relu(self.drop(self.conv_2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * NUM_CONV_2)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x
        # en utilisant loss = F.nll_loss(output, target) on peut faire
        # return F.log_softmax(x, dim=1)
