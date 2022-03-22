# -*- coding: utf-8 -*-
"""
Finetuning allows to adapt the model to our database.
Two types of fine tuning :
    update all parameters
    update only the parameters of the last layer
"""
# pylint:disable=W0621,C0103,C0116,C0413,R0913,R0914,C0209
# %load_ext autoreload
# %autoreload 2
import logging

logging.basicConfig(level=logging.INFO)
import os
import sys

sys.path.append(os.path.abspath(".."))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch import optim
from torchvision import transforms

from src.data.digits_locations import DigitRecognitionDataset
from src.models.class_cnnnet import CNNNet


def get_model():
    """
    Import model
    """
    model = CNNNet()
    model.load_state_dict(torch.load("model_cnn.pth"))
    return model


trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# J'ai split en train, val et test à partir de la liste complète

train_set = DigitRecognitionDataset("train", transforms=trans)
val_set = DigitRecognitionDataset("val", transforms=trans)
test_set = DigitRecognitionDataset("test", transforms=trans)


# ### Etude de la distribution des données

# +
def distribution(train_set, test_set, val_set):
    """
    distribution of train, test, val
    """
    train = [labels[0].item() for imputs, labels in train_set]
    test = [labels[0].item() for imputs, labels in test_set]
    val = [labels[0].item() for imputs, labels in val_set]
    return (train, test, val)


results = distribution(train_set, test_set, val_set)

fig, axes = plt.subplots(3, 1, figsize=(15, 7), sharex=True)

pd.Series(results[0]).value_counts().sort_index().plot.bar(
    ax=axes[0], title=("Distribution Train set")
)
pd.Series(results[1]).value_counts().sort_index().plot.bar(
    ax=axes[1], title=("Distribution Test set")
)
pd.Series(results[2]).value_counts().sort_index().plot.bar(
    ax=axes[2], title=("Distribution Val set")
)
# -

# On définit les datasets et loaders pytorch à partir des listes d'images de train / val / test

# +
BATCH_SIZE = 100
train_loader = torch.utils.data.DataLoader(dataset=train_set, BATCH_SIZE=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_set, BATCH_SIZE=BATCH_SIZE, shuffle=False)
# -

# Je définis une fonction d'évaluation


def evaluate(model, dataset, device=torch.device("cpu")):
    """
    evaluate the model
    """
    model.eval()
    avg_loss = 0.0
    avg_accuracy = 0
    for data in dataset:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs.unsqueeze(0))

        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        n_correct = torch.sum(preds == labels)

        avg_loss += loss.item()
        avg_accuracy += n_correct
    return avg_loss / len(dataset), float(avg_accuracy) / len(dataset)


# Je définis une fonction classique d'entraînement d'un modèle


def train(
    model, loader_train, data_val, optimizer, criterion, n_epochs=10, device=torch.device("cpu")
):
    """
    Train the model
    """
    model.train()
    loss_val_prev = float("inf")
    for _ in range(n_epochs):  # à chaque epochs
        # print("EPOCH % i" % epoch)
        for i, data in enumerate(
            loader_train
        ):  # itère sur les minibatchs via le loader apprentissage
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(
                device
            )  # on passe les données sur CPU / GPU
            # print(inputs.shape) #100 1 28 28 batch size couleur et taille
            optimizer.zero_grad()  # on réinitialise les gradients
            outputs = model(inputs)  # on calcule l'output
            labels = labels.flatten()

            loss = criterion(outputs, labels)  # on calcule la loss
            # loss.requires_grad = True

            model.train(False)
            loss_val, accuracy = evaluate(model, data_val)
            model.train(True)
            print(
                "{} loss train: {:1.4f}\t val {:1.4f}\tAcc (val): {:.1%}".format(
                    i, loss.item(), loss_val, accuracy
                )
            )

            loss.backward()  # on effectue la backprop pour calculer les gradients
            optimizer.step()  # on update les gradients en fonction des paramètres
            if loss_val_prev <= loss_val:
                return
            loss_val_prev = loss_val


# ## Fine tuning

# ### Tous les paramètres


def optimizer_all_fct(model):
    """
    opitmize all function
    """
    params_to_update = model.parameters()
    optimizer_all = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    return optimizer_all


# ### Seulement la dernière couche


def optimizer_only_last_fct(model):
    """
    optimizer only last function
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc_2.parameters():
        param.requires_grad = True
    optimizer_only_last = optim.SGD(model.fc_2.parameters(), lr=0.001, momentum=0.9)
    return optimizer_only_last


# #Uncomment to select only the last layer

# list_of_layers_to_finetune=['fc_2','fc.bias','layer4.1.conv2.weight',
# 'layer4.1.bn2.bias','layer4.1.bn2.weight']
# def optimizer_only_last_fct(model, list_of_layers_to_finetune) :
#     params_to_update=[]
#     for name,param in model.named_parameters():
#         if name in list_of_layers_to_finetune:
#             print("fine tune ",name)
#             params_to_update.append(param)
#             param.requires_grad = True
#         else:
#             param.requires_grad = False
#     optimizer_selected = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
#     return(optimizer_selected)


def remove_diag(A):
    removed = A[~np.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], int(A.shape[0]) - 1, -1)
    return np.squeeze(removed)


# ## Entrainement du modèle

# +
criterion = torch.nn.CrossEntropyLoss()


def train_model(model, train_loader, val_set, optimizer, criterion, n_epochs):
    torch.manual_seed(42)
    train(model, train_loader, val_set, optimizer, criterion, n_epochs)


def evaluate_model(model, type_set, train=True):
    _, accuracy = evaluate(model, type_set)
    print(f"Accuracy {'train' if train else 'test'}: %.1f%%" % (100 * accuracy))


def confusion_matrix_function():
    y_test = []
    y_test_pred = []
    for data in test_set:
        imputs, labels = data
        y_test.append(labels[0].numpy())
        pred = model(imputs.unsqueeze(0))
        pred_num = torch.argmax(pred)
        y_test_pred.append(pred_num.numpy())
    matrix = confusion_matrix(y_test, y_test_pred, normalize="true")
    return (matrix, classification_report(y_test, y_test_pred))


optimizer_functions = [optimizer_only_last_fct, optimizer_all_fct]

for optimizer_function in optimizer_functions:
    model = get_model()
    optimizer = optimizer_function(model)
    train_model(model, train_loader, val_set, optimizer, criterion, n_epochs=100)
    evaluate_model(model, train_set)
    evaluate_model(model, test_set, False)
    matrix, class_report = confusion_matrix_function()
    print(class_report)
    # matrix = remove_diag(matrix)
    ConfusionMatrixDisplay(matrix).plot()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

# +
# Sauvegarder le modèle entrainé sur optim_all

model = get_model()
optimizer = optimizer_all_fct(model)
train_model(model, train_loader, val_set, optimizer, criterion, n_epochs=100)
filename = "finetune_OCR.pth"
torch.save(model.state_dict(), filename)
print(f"saved model to {filename}")
