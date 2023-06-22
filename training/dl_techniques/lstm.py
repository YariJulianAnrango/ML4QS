import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
import time



train = pd.read_csv('/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/training/train.csv', index_col=0).reset_index(drop=True)
val = pd.read_csv('/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/training/val.csv', index_col=0).reset_index(drop=True)
test = pd.read_csv('/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/training/test.csv', index_col=0).reset_index(drop=True)


train_X = train.drop('combined', axis = 1)
train_y = train['combined']


val_X = val.drop('combined', axis = 1)
val_y = val[['combined']]


test_X = test.drop('combined', axis = 1)
test_y = test[['combined']]

label_encoder = LabelEncoder()
train_y = label_encoder.fit_transform(train_y)
val_y['combined'] = label_encoder.fit_transform(val_y['combined'])

inputs = torch.tensor(train_X.values, dtype=torch.float32)
labels = torch.tensor(train_y, dtype=torch.long)


class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, lstm_layers):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size


        self.linear1 = nn.Linear(hidden_size, num_classes)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers = lstm_layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.relu(x)
        x = self.linear1(x)
        return x



input_size = inputs.size(-1)
hidden_size = 256
num_classes = 11
lstm_layers = 2
model = LSTMNetwork(input_size, hidden_size, num_classes, lstm_layers)
model.to('mps')
inputs = inputs.to('mps')
labels = labels.to('mps')
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
start = time.time()
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)  # inputs is your input data

    # Compute loss
    loss = criterion(outputs, labels)  # labels is your target labels

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss for monitoring progress
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

end = time.time()
print('time:',end-start)