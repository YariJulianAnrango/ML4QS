from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import plot_tree



def Stratified_Split(New_weather_steps):
    New_weather_steps = pd.read_csv(New_weather_steps)
    New_weather_steps = New_weather_steps.drop("Unnamed: 0",axis=1)
    X = New_weather_steps.drop(columns=['combined', 'start', 'end'])
    y = New_weather_steps['combined']  
    train_X, test_X, train_y, test_y= train_test_split(X, y, test_size=0.2, random_state=42)
                                                       
    #VALIDATIONSET
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, stratify=train_y, random_state=42)
    print("train/test set created, using stratified_split")
    return(train_X, test_X, train_y, test_y, val_X, val_y)

def kfold_crossvalidation(New_weather_steps):
    New_weather_steps = pd.read_csv(New_weather_steps)
    New_weather_steps = New_weather_steps.drop("Unnamed: 0",axis=1)
    X = New_weather_steps.drop(columns=['combined', 'start', 'end'])
    y = New_weather_steps['combined']  
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for train_index , test_index in kf.split(X):
        train_X , test_X = X.iloc[train_index,:],X.iloc[test_index,:]
        train_y , test_y = y[train_index] , y[test_index]
    
    #VALIDATIONSET
    val_X = test_X
    val_y = test_y
    print("train/test set created, using kfold_crossvalidation")
    return(train_X, test_X, train_y, test_y, val_X, val_y)



def k_nearest_neighbor(train_X, train_y, val_X, val_y, n_neighbors=80, print_model_details=False):
    # Create the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    print("KNN created")
    knn.fit(train_X, train_y.values.ravel())
    print("model fitted")

    # Apply the model
    pred_prob_training_y = knn.predict_proba(train_X)
    pred_prob_test_y = knn.predict_proba(val_X)
    pred_training_y = knn.predict(train_X)
    pred_test_y = knn.predict(val_X)
    frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=knn.classes_)
    #frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=knn.classes_)
    print("predictions are made")

    print("F1-score:",f1_score(val_y, pred_test_y, average='weighted'))

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

    return pred_training_y, pred_test_y, frame_prob_training_y, pred_prob_test_y

def naive_bayes(train_X, train_y, val_X):
    # Create the model
    nb = GaussianNB()
    print("naive bayes created")
    train_y = train_y.values.ravel()
    # Fit the model
    nb.fit(train_X, train_y)
    print("model fitted")
    # Apply the model
    pred_prob_training_y = nb.predict_proba(train_X)
    pred_prob_test_y = nb.predict_proba(val_X)
    pred_training_y = nb.predict(train_X)
    pred_test_y = nb.predict(val_X)
    frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nb.classes_)
    frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nb.classes_)

    print("F1-score:",f1_score(val_y, pred_test_y, average='weighted'))


    return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

def random_forest(train_X, train_y, val_X, n_estimators=210, min_samples_leaf=5, criterion='gini', print_model_details=False, gridsearch=False):

    rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, criterion=criterion)
    print("Random forest model created")
    rf.fit(train_X, train_y.values.ravel())
    print("the model is fitted")

    pred_prob_training_y = rf.predict_proba(train_X)
    pred_prob_test_y = rf.predict_proba(val_X)
    pred_training_y = rf.predict(train_X)
    pred_test_y = rf.predict(val_X)
    frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=rf.classes_)
    frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=rf.classes_)
    print("predictions are made")
    print("F1-score:",f1_score(val_y, pred_test_y, average='weighted'))
    
    
    fig = plt.figure(figsize=(5, 5))
    plot_tree(rf.estimators_[0], 
                   feature_names=train_X.columns,  
                   filled=True,  
                   max_depth=4, 
                   impurity=False, 
                   proportion=True);
    fig.savefig('randomforest.png')

    #plot
    #parameter tuning
    param_dist = {'n_estimators': randint(0,200),
              'max_depth': randint(1,20)}
    rf = RandomForestClassifier()
    rand_search = RandomizedSearchCV(rf, 
                                    param_distributions = param_dist, 
                                    n_iter=5, 
                                    cv=5)
    rand_search.fit(train_X, train_y)
    print('Best hyperparameters:',  rand_search.best_params_)

    return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

def support_vector_machine_without_kernel(train_X, train_y, val_X, C=1, tol=1e-3, max_iter=1000, gridsearch=True, print_model_details=False):
        # Create the model
        if gridsearch:
            tuned_parameters = [{'max_iter': [1000, 2000], 'tol': [1e-3, 1e-4],
                         'C': [1, 10, 100]}]
            svm = GridSearchCV(LinearSVC(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            svm = LinearSVC(C=C, tol=tol, max_iter=max_iter)

        # Fit the model
        svm.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(svm.best_params_)

        if gridsearch:
            svm = svm.best_estimator_

        # Apply the model

        distance_training_platt = 1/(1+np.exp(svm.decision_function(train_X)))
        pred_prob_training_y = distance_training_platt / distance_training_platt.sum(axis=1)[:,None]
        distance_test_platt = 1/(1+np.exp(svm.decision_function(val_X)))
        pred_prob_test_y = distance_test_platt / distance_test_platt.sum(axis=1)[:,None]
        pred_training_y = svm.predict(train_X)
        pred_test_y = svm.predict(val_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=svm.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=svm.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y



def feedforward_neural_network(train_X, train_y, val_X, hidden_layer_sizes=(150,100,50), max_iter=300, activation='relu', solver = 'adam'):
    
    nn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter, solver = solver)

    # Fit the model
    nn.fit(train_X, train_y.values.ravel())

    # Apply the model
    pred_prob_training_y = nn.predict_proba(train_X)
    pred_prob_test_y = nn.predict_proba(val_X)
    pred_training_y = nn.predict(train_X)
    pred_test_y = nn.predict(val_X)
    frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nn.classes_)
    frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nn.classes_)

    #plot
    plt.plot(nn.loss_curve_)
    plt.title("Loss Curve", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

    print("plot showed")

    #parameter tuning
    param_grid = {
        'hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30)],
        'max_iter': [50, 100, 150],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
        }
    print("param_grid made")
    grid = GridSearchCV(nn, param_grid, n_jobs= -1, cv=5)
    print("grid made")
    grid.fit(train_X, train_y)

    print(grid.best_params_) 

    return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

split_method = "kfold_crossvalidation"
if split_method == "Stratified_Split":
     train_X, test_X, train_y, test_y, val_X, val_y = Stratified_Split("C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\missing_weather_steps.csv")
elif split_method == "kfold_crossvalidation":
     train_X, test_X, train_y, test_y, val_X, val_y = kfold_crossvalidation("C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\missing_weather_steps.csv")
else:
    print("done")

algorithm = "k_nearest_neighbor"
if algorithm == "k_nearest_neighbor":
    pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y = k_nearest_neighbor(train_X, train_y, val_X, val_y, n_neighbors=80, print_model_details=False)
elif algorithm == "support_vector_machine_without_kernel":
    pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y = support_vector_machine_without_kernel(train_X, train_y, val_X, C=1, tol=1e-3, max_iter=100, gridsearch=True, print_model_details=False)
elif algorithm == "naive_bayes":
    pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y = naive_bayes(train_X, train_y, val_X)
elif algorithm == "random_forest":
    pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y = random_forest(train_X, train_y, val_X, n_estimators=10, min_samples_leaf=5, criterion='gini', print_model_details=False, gridsearch=False)
elif algorithm == "feedforward_neural_network":
    pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y = feedforward_neural_network(train_X, train_y, val_X, hidden_layer_sizes=(150,100,50), max_iter=300, activation='relu', solver = 'adam')
else:
    print("done")

#accuracy =  accuracy_score(frame_prob_test_y, test_y)
#print("Accuracy:", accuracy)



