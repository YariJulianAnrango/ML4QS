from sklearn.model_selection import train_test_split
import pandas as pd

def Stratified_Split(New_weather_steps):
    X = New_weather_steps.drop(columns=['combined', 'Unnamed: 0.1', 'Unnamed: 0'])
    y = New_weather_steps['combined']

    # Test/train
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=13, stratify=y)

    # validation/train
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, stratify=train_y,
                                                      random_state=13)

    print("train/validation/test set created, using stratified_split")
    return train_X, train_y, val_X, val_y, test_X, test_y



def temporal_split(df, test_date = '2023-01-01'):
    train = df[pd.to_datetime(df['start']) < pd.to_datetime(test_date)]
    test_val = df[pd.to_datetime(df['start']) >= pd.to_datetime(test_date)].reset_index(drop=True)
    test = test_val.loc[:round(len(test_val)/2),:].reset_index(drop=True)
    val = test_val.loc[round(len(test_val)/2):,:].reset_index(drop=True)

    return train, test, val
