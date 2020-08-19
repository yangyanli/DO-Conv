from pathlib import Path
import requests
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset
from torch import nn
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from torch.nn import Conv2d
from do_conv_pytorch import DOConv2d

random.seed(2020)
np.random.seed(2020)
torch.manual_seed(2020)

print(torch.cuda.is_available())
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")



x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()


bs = 32  # batch size
lr = 0.01
epochs = 15  # how many epochs to train for


train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)


loss_func = F.cross_entropy

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)



def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    i = 0
    for epoch in range(epochs):
        model.train()

        for xb, yb in train_dl:
            i = i+1
            loss, _ = loss_batch(model, loss_func, xb, yb, opt)
            if i % 1000 == 0:
                print('loss of %d' %i, 'step: %.4f' %loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in valid_dl:
                # images, labels = data
                xb, yb = xb.to(dev), yb.to(dev)
                outputs = model(xb)
                _, predicted = torch.max(outputs.data, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()

        print('Accuracy of the network on the test set: %.2f %%' % (
                100 * correct / total))



def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

############  Conv2d  #################################
model = nn.Sequential(
    Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1)),
)
########################################################

############  DO-Conv2d  ###############################
# model = nn.Sequential(
#     DOConv2d(1, 16, kernel_size=3, stride=2, padding=1),
#     nn.ReLU(),
#     DOConv2d(16, 16, kernel_size=3, stride=2, padding=1),
#     nn.ReLU(),
#     DOConv2d(16, 10, kernel_size=3, stride=2, padding=1),
#     nn.ReLU(),
#     nn.AdaptiveAvgPool2d(1),
#     Lambda(lambda x: x.view(x.size(0), -1)),
# )
########################################################

model.to(dev)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
fit(epochs, model, loss_func, opt, train_dl, valid_dl)







