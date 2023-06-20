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
    print("train/test set created, using Stratified_Split")                                            
    return(train_X, test_X, train_y, test_y)

def kfold_crossvalidation(New_weather_steps):
    New_weather_steps = pd.read_csv(New_weather_steps)
    #New_weather_steps = New_weather_steps.drop("Unnamed: 0",axis=1)
    X = X = New_weather_steps.drop(columns=['combined', 'Unnamed: 0.1', 'Unnamed: 0'])
    y = New_weather_steps['combined']  
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=13)
    for train_index , test_index in kf.split(X):
        train_X , test_X = X.iloc[train_index,:],X.iloc[test_index,:]
        train_y , test_y = y[train_index] , y[test_index]
    
    print("train/test set created, using kfold_crossvalidation")
    return(train_X, test_X, train_y, test_y)

def k_nearest_neighbor(train_X, train_y, test_X, test_y, n_neighbors=9):
    # Create the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    print("KNN created")
    knn.fit(train_X, train_y.values.ravel())
    print("model fitted")

    # Apply the model
    pred_test_y = knn.predict(test_X)
    #frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=knn.classes_)


    print("F1-score:",f1_score(test_y, pred_test_y, average='weighted'))
    print("recall_score:",recall_score(test_y, pred_test_y, average='weighted'))
    print("precision_score:",precision_score(test_y, pred_test_y, average='weighted'))

def naive_bayes(train_X, train_y, test_X, test_y, var_smoothing=1.0):
    # Create the model
    nb = GaussianNB(var_smoothing= var_smoothing)
    print("naive bayes created")
    train_y = train_y.values.ravel()
    # Fit the model
    nb.fit(train_X, train_y)
    print("model fitted")
    # Apply the model
    pred_test_y = nb.predict(test_X)

    print("F1-score:",f1_score(test_y, pred_test_y, average='weighted'))
    print("recall_score:",recall_score(test_y, pred_test_y, average='weighted'))
    print("precision_score:",precision_score(test_y, pred_test_y, average='weighted'))
    

def random_forest(train_X, train_y, test_X, test_y, n_estimators=269, min_samples_leaf=10, criterion='gini', max_depth= 38):

    rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, criterion=criterion, max_depth = max_depth)
    print("Random forest model created")
    rf.fit(train_X, train_y.values.ravel())
    print("the model is fitted")

    pred_test_y = rf.predict(test_X)
    
    print("F1-score:",f1_score(test_y, pred_test_y, average='weighted'))
    print("recall_score:",recall_score(test_y, pred_test_y, average='weighted'))
    print("precision_score:",precision_score(test_y, pred_test_y, average='weighted'))
    


split_method = "Stratified_Split"
if split_method == "Stratified_Split":
     train_X, test_X, train_y, test_y= Stratified_Split("C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\features_selected_new.csv")
elif split_method == "kfold_crossvalidation":
     train_X, test_X, train_y, test_y= kfold_crossvalidation("C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\features_selected_new.csv")
else:
    print("done")

algorithm = "naive_bayes"
if algorithm == "k_nearest_neighbor":
    k_nearest_neighbor(train_X, train_y, test_X, test_y, n_neighbors=9)
elif algorithm == "naive_bayes":
    naive_bayes(train_X, train_y, test_X, test_y, var_smoothing=1.0)
elif algorithm == "random_forest":
    random_forest(train_X, train_y, test_X, test_y,  n_estimators=269, min_samples_leaf=10, criterion='gini', max_depth= 38)
else:
    print("done")