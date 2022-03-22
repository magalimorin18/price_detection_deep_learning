"""
Train a CNN model on the MNIST dataset
"""
# pylint: disable = W0621
import torch
import torch.nn.functional as F
from torch import nn

# import datasets
from torchvision import datasets, transforms

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device = {}".format(device))

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_set = datasets.MNIST("./data", train=True, transform=trans, download=True)
test_set = datasets.MNIST("./data", train=False, transform=trans, download=True)

# define data loaders
BATCH_SIZE = 100
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)
# print("total training batch number: {}".format(len(train_loader)))
# print("total testing batch number: {}".format(len(test_loader)))


# +

DATA_SIZE = 784
NUM_HIDDEN_1 = 256  # try 512
NUM_HIDDEN_2 = 256
NUM_CLASSES = 10

NUM_CONV_1 = 10
NUM_CONV_2 = 20
NUM_FC = 500


class CNNNet(nn.Module):
    """
    Class CNN
    """

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, NUM_CONV_1, 5, 1)
        self.conv_2 = nn.Conv2d(NUM_CONV_1, NUM_CONV_2, 5, 1)
        self.fc_1 = nn.Linear(4 * 4 * NUM_CONV_2, NUM_FC)
        self.fc_2 = nn.Linear(NUM_FC, NUM_CLASSES)

    def forward(self, x):
        """
        Function forward
        """
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * NUM_CONV_2)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x


# define model
model = CNNNet()
IS_CNN = True
model.to(device)  # puts model on GPU / CPU

# +

# optimization hyperparameters
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
loss_fn = nn.CrossEntropyLoss()

# main loop (train+test)
for epoch in range(10):
    # training
    model.train()  # mode "train" agit sur "dropout" ou "batchnorm"
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x, target = x.to(device), target.to(device)
        out = model(x)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f"epoch {epoch} batch {batch_idx}"
                f"[{batch_idx * len(x)}/{len(train_loader.dataset)}] training loss: {loss.item()}"
            )

    # testing
    model.eval()
    CORRECT = 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_loader):
            x, target = x.to(device), target.to(device)
            out = model(x)
            loss = loss_fn(out, target)
            # _, prediction = torch.max(out.data, 1)
            prediction = out.argmax(dim=1, keepdim=True)  # index of the max log-probability
            CORRECT += prediction.eq(target.view_as(prediction)).sum().item()
    taux_classif = 100.0 * CORRECT / len(test_loader.dataset)
    print(
        f"Accuracy: {CORRECT}/{len(test_loader.dataset)}"
        f"(tx {taux_classif}%, err {100.0 - taux_classif}%)\n"
    )

# +

if IS_CNN is True:
    FILENAME = "model_cnn.pth"
    torch.save(model.state_dict(), FILENAME)
    print(f"saved model to {FILENAME}")
