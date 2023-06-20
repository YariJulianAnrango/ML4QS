import pandas as pd
feature_selected = pd.read_csv("C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\features_selected_new.csv")
feature_selected = feature_selected.dropna()
feature_selected.to_csv('features_selected_new.csv')
