"""Digit classifier."""
# pylint: disable=wrong-import-position
# %%
import os
import sys

import matplotlib.pyplot as plt
import torch
from cv2 import transform
from torch import nn
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join("..", "..")))
from src.config import SAVED_MODELS

# %%
# Define the transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


# %%
mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)


# %%
model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(1568, 10),
)

# %%
# Check if the model is working
test_image = mnist_train[0][0]
plt.imshow(test_image.numpy().squeeze(), cmap="gray")
print(f"Test image shape: {test_image.shape}")
with torch.no_grad():
    model.eval()
    print(test_image.dtype, type(test_image))
    output = model(test_image.unsqueeze(0))
    print(f"Model output shape: {output.shape}")
    print(output)
# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
EPOCHS = 2
BATCH_SIZE = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)

model.to(DEVICE)

for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch}, batch: {batch_idx}, loss: {loss}")

# %%
torch.save(model.state_dict(), os.path.join(SAVED_MODELS, "digit_classifier.pt"))
