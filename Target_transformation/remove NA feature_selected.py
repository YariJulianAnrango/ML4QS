import pandas as pd
feature_selected = pd.read_csv("C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\feature_selected.csv")
feature_selected = feature_selected.dropna()
feature_selected.to_csv('feature_selected.csv')
