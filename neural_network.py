import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import data

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Neural_Network(nn.Module):
    def __init__(self, input_size, hidden_size, nb_layers, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.nb_layers = nb_layers

        self.lstm = nn.LSTM(input_size, hidden_size, nb_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.nb_layers, x.size, self.hidden_size)
        c0 = torch.zeros(self.nb_layers, x.size, self.hidden_size)

        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

def pad_tensors(tensors_list):
    max_shape = max([t.shape for t in tensors_list], key=lambda x: x[1])
    padded_tensors = []
    for tensor in tensors_list:
        padding = torch.zeros((tensor.shape[0], max_shape[1] - tensor.shape[1], tensor.shape[2]), dtype=torch.float32)
        padded_tensor = torch.cat((tensor, padding), dim=1)
        padded_tensors.append(padded_tensor)
    return padded_tensors


def train(model, train_data, train_labels, optimizer, criterion, epochs):
    padded_train_data = pad_tensors(train_data)
    padded_train_labels = pad_tensors(train_labels)
    train_data = torch.stack(padded_train_data, dim=0)
    train_labels = torch.stack(padded_train_labels, dim=0)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_data)
        loss = criterion(output, train_labels)
        loss.backward()
        optimizer.step()

        # model.eval()
        # with torch.no_grad():
        #     test_output = model(test_data)
        #     test_loss = criterion(test_output, test_labels)
        if epoch % 100 == 0:
            print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
        # if test_loss > loss:
        #     break


def test(model, test_data, test_labels):
    model.eval()
    with torch.no_grad():
        output = model(test_data)
        loss = criterion(output, test_labels)
    print('Test Loss: {:.4f}'.format(loss.item()))


input_size = 29
hidden_size = 128
nb_layers = 2
output_size = 29
epochs = 2
learning_rate = 0.001

model = Neural_Network(input_size, hidden_size, nb_layers, output_size)
# torch.load(model.state_dict(), 'model.pth')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

alphabet = "' abcdefghijklmnopqrstuvwxyz"
db = data.DataBase('dataset/fr/', alphabet, limit=100)
print("DONE !")
train(model, db.data['data'], db.data['label'], optimizer, criterion, epochs)
torch.save(model.state_dict(), 'model.pth')
# test(model, test_data, test_labels)
