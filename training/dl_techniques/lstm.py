import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np


class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size,  lstm_layers, num_classes = 12):
        super(LSTMNetwork, self).__init__()
        self.lstm_layers = lstm_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=lstm_layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        print(x.size())
        h0 = torch.zeros(self.lstm_layers, x.size(1), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm_layers, x.size(1), self.hidden_size).to(x.device)
        x, _ = self.lstm(x,(h0, c0))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear1(x)
        return x.to(torch.float32)


def train_lstm(train_X, train_y, num_epochs, m1=False, hidden_size=256, num_classes=12, lstm_layers=2):
    label_encoder = LabelEncoder()
    train_y = label_encoder.fit_transform(train_y)

    inputs = torch.tensor(train_X.values, dtype=torch.float32)
    inputs = inputs.unsqueeze(0).transpose(0, 1)
    labels = torch.tensor(train_y, dtype=torch.float32)
    unique_labels = torch.unique(labels)
    print(unique_labels)

    input_size = inputs.size(-1)
    model = LSTMNetwork(input_size, hidden_size, num_classes, lstm_layers)

    if m1 == True:
        model.to('mps')
        inputs = inputs.to('mps')
        labels = labels.to('mps')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    start = time.time()
    loss_epochs = []
    for epoch in range(num_epochs):
        outputs = model(inputs)
        outputs = outputs.squeeze()
        print(outputs.size())
        print(labels.size())
        loss = criterion(outputs, labels)

        loss_epochs.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    end = time.time()
    print('time:', end - start)

    path = '/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/models/lstm_' + str(
        pd.Timestamp.now().strftime('%Y-%m-%d_%H:%M:%S')) + '.pt'
    torch.save(model.state_dict(), path)
    print(loss_epochs)
    plt.plot(range(len(loss_epochs)), loss_epochs)
    plt.title('Cross entropy loss of LSTM models per epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('plot1')


def load_lstm(dat_X, model_dir, hidden_size=256, num_classes=12, lstm_layers=1):
    inputs = torch.tensor(dat_X.values, dtype=torch.float32)

    input_size = inputs.size(-1)

    model = LSTMNetwork(input_size, hidden_size, num_classes, lstm_layers)
    model.load_state_dict(torch.load(
        model_dir))

    return model

def optimize_hyperparameters(train_X,
                             train_y,
                             hidden_size = [50, 256],
                             lstm_layers = [1,2]):
    label_encoder = LabelEncoder()
    train_y = label_encoder.fit_transform(train_y)

    inputs = torch.tensor(train_X.values, dtype=torch.float32)
    labels = torch.tensor(train_y, dtype=torch.long)

    input_size = inputs.size(-1)

    model = NeuralNetClassifier(
        LSTMNetwork,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        verbose=2,
        module__input_size = input_size,
        max_epochs = 100,
        optimizer__lr = 0.01
    )
    param_grid = {
        'module__hidden_size': hidden_size,
        'module__lstm_layers':lstm_layers
    }
    start = time.time()
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(inputs, labels)
    end = time.time()
    print('time:', end-start)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid_result
