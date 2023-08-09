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
    

def random_forest(train_X, train_y, val_X, val_y, n_estimators, min_samples_leaf, criterion, max_depth):

    rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, criterion=criterion, max_depth = max_depth)
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
    k_nearest_neighbor(train_X, train_y, val_X, val_y, n_neighbors=42)
elif algorithm == "naive_bayes":
    naive_bayes(train_X, train_y, val_X, val_y, var_smoothing=0.2848035868435802)
elif algorithm == "random_forest":
    random_forest(train_X, train_y, val_X, val_y,  n_estimators=194, min_samples_leaf=2, criterion='gini', max_depth= 24)
else:
    print("done")