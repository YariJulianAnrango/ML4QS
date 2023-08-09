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

#In dit bestand wordt parameter tuning gedaan van KNN, naive bayes en Random forest
#De validation en train split zijn al gemaakt in val.csv en train.csv. De functie split zorgt ervoor dat deze csv files
#omgezet kunnen worden naar bruikbare data

def Split(val_csv, train_csv):
    val = pd.read_csv(val_csv)
    val_X = val.drop(columns=['combined', 'Unnamed: 0'])
    val_y = val['combined']
    train = pd.read_csv(train_csv)
    train_X = train.drop(columns=['combined', 'Unnamed: 0'])
    train_y = train['combined']
    return(train_X, train_y, val_X, val_y)

#Bij het kijken naar de parameter K van KNN wordt een plot gemaakt die de error_rate weergeeft. Daar waar de error_rate
#Het laagst is, dat is de beste K
def k_nearest_neighbor(train_X, train_y, val_X, val_y):
    #plot
    error_rate = []
    # searching k value upto  100
    for i in range(1,100):
        print(i)
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

#Bij naive bayes wordt gekeken naar de parameter var_smoothing. Die wordt geoptimaliseerd door te kijken naar 
#de parameter grid
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


#Bij deze werkt het iets anders, omdat ik de code niet zo kon krijgen dat hij daadwerkelijk trained net zoals 
#bij KNN en Naive bayes. Op internet vond ik ook niks, dus heb ik het zo gedaan.
#Hij genereerd telkens random waardes voor N_estimator, min_sample_leaf en max_depth. Daarbij rekent hij de f1-score
#Na 100 runs kijk je gewoon in het lijstje wat geprint is en welke combinatie de hoogste f1 score krijgt.
def random_forest(train_X, train_y, val_X, val_y):
    
    # parameter tuning
    error_rate = []
    
    param_grid = {'n_estimators': randint(50, 300),
                    'min_samples_leaf':  randint(1, 20),
                    'max_depth': randint(20, 50)}
    # naive algorithm 
    for i in range(0,100):
        print(i)
        grid_search = RandomizedSearchCV(RandomForestClassifier(), param_grid, n_iter=1, cv=5)
        #print("grid search made")
        grid_search.fit(train_X, train_y.values.ravel())
        #print("model fitted")
        # testing the model
        pred_i = grid_search.predict(val_X)
        print("F1-score:",f1_score(val_y, pred_i, average='weighted'))
        print(grid_search.best_params_)
        error_rate.append(np.mean(pred_i != val_y))
    
    #Niet te veel aandacht leggen op de plot, want die heb ik niet meer gebruikt
    # Configure and plot error rate over k values
    plt.figure(figsize=(10, 101))
    plt.plot(range(1, 101), error_rate, color='blue', linestyle='dashed', markersize=10)
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

train_X, train_y, val_X, val_y = Split("C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\training\\val.csv", "C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\training\\train.csv")

algorithm = "naive_bayes"
if algorithm == "k_nearest_neighbor":
    k_nearest_neighbor(train_X, train_y, val_X, val_y)
elif algorithm == "naive_bayes":
    naive_bayes(train_X, train_y, val_X, val_y)
elif algorithm == "random_forest":
    random_forest(train_X, train_y, val_X, val_y)
else:
    print("done")





