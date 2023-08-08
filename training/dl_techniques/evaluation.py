import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from training.dl_techniques.lstm import load_lstm
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

val = pd.read_csv('/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/training/val.csv', index_col=0).reset_index(drop=True)
test = pd.read_csv('/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/training/test.csv', index_col=0).reset_index(drop=True)

val_X = val.drop('combined', axis = 1)
val_y = val[['combined']]

test_X = test.drop('combined', axis = 1)
test_y = test[['combined']]

label_encoder = LabelEncoder()
val_y = label_encoder.fit_transform(val_y)
test_y = label_encoder.fit_transform(test_y)

model = load_lstm(val_X, model_dir='/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/models/lstm_2023-06-25_20:37:39.pt')
# Evaluation
model.eval()

inputs = torch.tensor(val_X.values, dtype=torch.float32)
labels = torch.tensor(val_y, dtype=torch.long)

outputs = model(inputs)
_, preds = torch.max(outputs, 1)
preds = preds.cpu().numpy()
classes = labels.cpu().numpy()

score = f1_score(classes, preds, average='weighted')
acc = accuracy_score(classes, preds)
print(score)
print(acc)
