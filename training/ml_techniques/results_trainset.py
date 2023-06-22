from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import plot_tree



def Stratified_Split(New_weather_steps):
    New_weather_steps = pd.read_csv(New_weather_steps)
    #New_weather_steps = New_weather_steps.drop("Unnamed: 0",axis=1)
    X = New_weather_steps.drop(columns=['combined', 'Unnamed: 0.1', 'Unnamed: 0'])
    y = New_weather_steps['combined']  
    train_X, test_X, train_y, test_y= train_test_split(X, y, test_size=0.2, random_state=13)
                                                       
    #VALIDATIONSET
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, stratify=train_y, random_state=13)
    print("train/test set created, using stratified_split")
    return(train_X, train_y, val_X, val_y)

def kfold_crossvalidation(New_weather_steps):
    New_weather_steps = pd.read_csv(New_weather_steps)
    #New_weather_steps = New_weather_steps.drop("Unnamed: 0",axis=1)
    X = X = New_weather_steps.drop(columns=['combined', 'Unnamed: 0.1', 'Unnamed: 0'])
    y = New_weather_steps['combined']  
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=12)
    for train_index , test_index in kf.split(X):
        train_X , test_X = X.iloc[train_index,:],X.iloc[test_index,:]
        train_y , test_y = y[train_index] , y[test_index]
    
    #VALIDATIONSET
    val_X = test_X
    val_y = test_y
    print("train/test set created, using kfold_crossvalidation")
    return(train_X, train_y, val_X, val_y)

def Split(val_csv, train_csv):
    val = pd.read_csv(val_csv)
    val_X = val.drop(columns=['combined', 'Unnamed: 0'])
    val_y = val['combined']
    train = pd.read_csv(train_csv)
    train_X = train.drop(columns=['combined', 'Unnamed: 0'])
    train_y = train['combined']
    return(train_X, train_y, val_X, val_y)

def k_nearest_neighbor(train_X, train_y, val_X, val_y, n_neighbors):
    # Create the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    print("KNN created")
    knn.fit(train_X, train_y.values.ravel())
    print("model fitted")

    # Apply the model
    pred_val_y = knn.predict(val_X)
    #frame_prob_val_y = pd.DataFrame(pred_prob_val_y, columns=knn.classes_)


    print("F1-score:",f1_score(val_y, pred_val_y, average='weighted'))
    print("recall_score:",recall_score(val_y, pred_val_y, average='weighted'))
    print("precision_score:",precision_score(val_y, pred_val_y, average='weighted'))

def naive_bayes(train_X, train_y, val_X, val_y, var_smoothing):
    # Create the model
    nb = GaussianNB(var_smoothing= var_smoothing)
    print("naive bayes created")
    train_y = train_y.values.ravel()
    # Fit the model
    nb.fit(train_X, train_y)
    print("model fitted")
    # Apply the model
    pred_val_y = nb.predict(val_X)

    print("F1-score:",f1_score(val_y, pred_val_y, average='weighted'))
    print("recall_score:",recall_score(val_y, pred_val_y, average='weighted'))
    print("precision_score:",precision_score(val_y, pred_val_y, average='weighted'))
    

def random_forest(train_X, train_y, val_X, val_y, n_estimators, min_samples_leaf, max_depth, criterion, max_features):

    rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, criterion=criterion, max_depth = max_depth, max_features = max_features)
    print("Random forest model created")
    rf.fit(train_X, train_y.values.ravel())
    print("the model is fitted")

    pred_val_y = rf.predict(val_X)
    
    print("F1-score:",f1_score(val_y, pred_val_y, average='weighted'))
    print("recall_score:",recall_score(val_y, pred_val_y, average='weighted'))
    print("precision_score:",precision_score(val_y, pred_val_y, average='weighted'))
    


train_X, train_y, val_X, val_y = Split("C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\training\\val.csv", "C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\training\\train.csv")


algorithm = "random_forest"
if algorithm == "k_nearest_neighbor":
    k_nearest_neighbor(train_X, train_y, val_X, val_y, n_neighbors=20)
elif algorithm == "naive_bayes":
    naive_bayes(train_X, train_y, val_X, val_y, var_smoothing=1)
elif algorithm == "random_forest":
    random_forest(train_X, train_y, val_X, val_y,  n_estimators=27, min_samples_leaf=21, max_depth= 3, criterion='gini', max_features = 1)
else:
    print("done")