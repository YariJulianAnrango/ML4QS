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


def k_nearest_neighbor(train_X, train_y, val_X, val_y):
    #plot
    error_rate = []
    # searching k value upto  100
    for i in range(1,100):
        # knn algorithm 
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(train_X, train_y.values.ravel())
        # testing the model
        pred_i = knn.predict(val_X)
        error_rate.append(np.mean(pred_i != val_y))
    # Configure and plot error rate over k values
    plt.figure(figsize=(10,4))
    plt.plot(range(1,100), error_rate, color='blue', linestyle='dashed', markersize=10)
    plt.title('Error Rate vs. K-Values')
    plt.xlabel('K-Values')
    plt.ylabel('Error Rate')
    plt.savefig("line_plot.png")

def naive_bayes(train_X, train_y, val_X, val_y):
    #tuning
    nb = GaussianNB()
    param_grid_nb = {'var_smoothing': np.logspace(0,-9, num=100)}
    nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)
    nbModel_grid.fit(train_X, train_y)
    print(nbModel_grid.best_estimator_)
    #print(nb.class_prior_)

    error_rate = []
    for i in range(0,3):
        # naive algorithm 
        nb = GaussianNB(var_smoothing=i)
        nb.fit(train_X, train_y.values.ravel())
        # testing the model
        pred_i = nb.predict(val_X)
        error_rate.append(np.mean(pred_i != val_y))
    # Configure and plot error rate over k values
    plt.figure(figsize=(10,4))
    plt.plot(range(0,3), error_rate, color='blue', linestyle='dashed', markersize=10)
    plt.title('Error Rate vs. Naive Bayes')
    plt.xlabel('Naive Bayes')
    plt.ylabel('Error Rate')
    plt.savefig("Naive_bayes.png")

def random_forest(train_X, train_y, val_X, val_y):
    
    # parameter tuning
    error_rate = []
    
    param_grid = {'n_estimators': randint(50, 300),
                    'min_samples_leaf':  randint(1, 20),
                    'max_depth': randint(20, 50)}
    # naive algorithm 
    for i in range(0,50):
        grid_search = RandomizedSearchCV(RandomForestClassifier(), param_grid, n_iter=1, cv=5)
        print("grid search made")
        grid_search.fit(train_X, train_y.values.ravel())
        print("model fitted")
        # testing the model
        pred_i = grid_search.predict(val_X)
        print("predictions made")
        error_rate.append(np.mean(pred_i != val_y))
    
    # Configure and plot error rate over k values
    plt.figure(figsize=(10, 51))
    plt.plot(range(1, 51), error_rate, color='blue', linestyle='dashed', markersize=10)
    plt.title('Error Rate vs. Random forest')
    plt.xlabel('Random forest')
    plt.ylabel('Error Rate')
    plt.savefig("Random_forest.png")
    
    
    
    
    param_grid = {'n_estimators': [50,100],
                  'min_samples_leaf':  [50,100],
                  'max_depth': [50,100]}
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid_search = RandomizedSearchCV(RandomForestClassifier(), param_grid, n_iter=50 , cv=5)
    grid_search.fit(train_X, train_y.values.ravel())
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)

split_method = "kfold_crossvalidation"
if split_method == "Stratified_Split":
     train_X, train_y, val_X, val_y = Stratified_Split("C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\features_selected_new.csv")
elif split_method == "kfold_crossvalidation":
     train_X, train_y, val_X, val_y = kfold_crossvalidation("C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\features_selected_new.csv")
else:
    print("done")

algorithm = "random_forest"
if algorithm == "k_nearest_neighbor":
    k_nearest_neighbor(train_X, train_y, val_X, val_y)
elif algorithm == "naive_bayes":
    naive_bayes(train_X, train_y, val_X, val_y)
elif algorithm == "random_forest":
    random_forest(train_X, train_y, val_X, val_y)
else:
    print("done")





