from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




def Stratified_Split(New_weather_steps):
    New_weather_steps = pd.read_csv(New_weather_steps)
    New_weather_steps = New_weather_steps.drop("Unnamed: 0",axis=1)
    X = New_weather_steps.drop(columns=['combined', 'start', 'end'])
    y = New_weather_steps['combined']  
    train_X, test_X, train_y, test_y= train_test_split(X, y, test_size=0.2, random_state=42)
                                                       
    #VALIDATIONSET
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, stratify=train_y, random_state=42)
    #print("train/test set created, using stratified_split")
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
    #print("train/test set created, using kfold_crossvalidation")
    return(train_X, test_X, train_y, test_y, val_X, val_y)



def k_nearest_neighbor(train_X, train_y, val_X, val_y, n_neighbors=80, gridsearch=True, print_model_details=False):
    # Create the model
    if gridsearch:
        tuned_parameters = [{'n_neighbors': [1, 2, 5, 10]}]
        knn = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, scoring='accuracy')
    else:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    #print("KNN created")
    # Fit the model
    knn.fit(train_X, train_y.values.ravel())
    #print("model fitted")
    if gridsearch and print_model_details:
        print(knn.best_params_)

    if gridsearch:
        knn = knn.best_estimator_

    # Apply the model
    pred_prob_training_y = knn.predict_proba(train_X)
    pred_prob_test_y = knn.predict_proba(val_X)
    pred_training_y = knn.predict(train_X)
    pred_test_y = knn.predict(val_X)
    frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=knn.classes_)
    #frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=knn.classes_)
    #print("predictions are made")

    #print("Accuracy:",metrics.accuracy_score(val_y, pred_test_y))

    #plot
    k_values = [i for i in range (1,100)]
    scores = []

    scaler = StandardScaler()
    X = scaler.fit_transform(val_X)

    for k in k_values:
        #print(k)
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, X, val_y, cv=5)
        scores.append(np.mean(score))
    sns.lineplot(x = k_values, y = scores, marker = 'o')
    plt.xlabel("K Values")
    plt.ylabel("Accuracy Score")
    plt.title("Accuracy score of different K Values")
    plt.savefig("line_plot.png")

    return pred_training_y, pred_test_y, frame_prob_training_y, pred_prob_test_y

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

def naive_bayes(train_X, train_y, val_X):
        # Create the model
        nb = GaussianNB()
        
        train_y = train_y.values.ravel()
        # Fit the model
        nb.fit(train_X, train_y)

        # Apply the model
        pred_prob_training_y = nb.predict_proba(train_X)
        pred_prob_test_y = nb.predict_proba(val_X)
        pred_training_y = nb.predict(train_X)
        pred_test_y = nb.predict(val_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nb.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nb.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

def random_forest(train_X, train_y, val_X, n_estimators=10, min_samples_leaf=5, criterion='gini', print_model_details=False, gridsearch=True):

    if gridsearch:
        tuned_parameters = [{'min_samples_leaf': [2, 10, 50, 100, 200],
                                'n_estimators':[10, 50, 100],
                                'criterion':['gini', 'entropy']}]
        rf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='accuracy')
    else:
        rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, criterion=criterion)
    print("Random forest model created")
    
    # Fit the model
    rf.fit(train_X, train_y.values.ravel())
    print("the model is fitted")

    if gridsearch and print_model_details:
        print(rf.best_params_)

    if gridsearch:
        rf = rf.best_estimator_

    pred_prob_training_y = rf.predict_proba(train_X)
    pred_prob_test_y = rf.predict_proba(val_X)
    pred_training_y = rf.predict(train_X)
    pred_test_y = rf.predict(val_X)
    frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=rf.classes_)
    frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=rf.classes_)
    print("predictions are made")

    if print_model_details:
        ordered_indices = [i[0] for i in sorted(enumerate(rf.feature_importances_), key=lambda x:x[1], reverse=True)]
        print('Feature importance random forest:')
        for i in range(0, len(rf.feature_importances_)):
            print(train_X.columns[ordered_indices[i]], end='')
            print(' & ', end='')
            print(rf.feature_importances_[ordered_indices[i]])
    print("predicted variables made")
    return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

def feedforward_neural_network(train_X, train_y, val_X, hidden_layer_sizes=(100,), max_iter=500, activation='logistic', alpha=0.0001, learning_rate='adaptive', gridsearch=True, print_model_details=False):
    if gridsearch:
        # With the current parameters for max_iter and Python 3 packages convergence is not always reached, therefore increased +1000.
        tuned_parameters = [{'hidden_layer_sizes': [(5,), (10,), (25,), (100,), (100,5,), (100,10,),], 'activation': [activation],
                                'learning_rate': [learning_rate], 'max_iter': [2000, 3000], 'alpha': [alpha]}]
        nn = GridSearchCV(MLPClassifier(), tuned_parameters, cv=5, scoring='accuracy')
    else:
        # Create the model
        nn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter, learning_rate=learning_rate, alpha=alpha, random_state=42)

    # Fit the model
    nn.fit(train_X, train_y.values.ravel())

    if gridsearch and print_model_details:
        print(nn.best_params_)

    if gridsearch:
        nn = nn.best_estimator_

    # Apply the model
    pred_prob_training_y = nn.predict_proba(train_X)
    pred_prob_test_y = nn.predict_proba(val_X)
    pred_training_y = nn.predict(train_X)
    pred_test_y = nn.predict(val_X)
    frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nn.classes_)
    frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nn.classes_)

    return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

split_method = "kfold_crossvalidation"
if split_method == "Stratified_Split":
     train_X, test_X, train_y, test_y, val_X, val_y = Stratified_Split("C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\Missing_weather_steps.csv")
elif split_method == "kfold_crossvalidation":
     train_X, test_X, train_y, test_y, val_X, val_y = kfold_crossvalidation("C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\Missing_weather_steps.csv")
else:
    print("done")

algorithm = "k_nearest_neighbor"
if algorithm == "k_nearest_neighbor":
    pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y = k_nearest_neighbor(train_X, train_y, val_X, val_y, n_neighbors=80, gridsearch=False, print_model_details=False)
elif algorithm == "support_vector_machine_without_kernel":
    pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y = support_vector_machine_without_kernel(train_X, train_y, val_X, C=1, tol=1e-3, max_iter=100, gridsearch=True, print_model_details=False)
elif algorithm == "naive_bayes":
    pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y = naive_bayes(train_X, train_y, val_X)
elif algorithm == "random_forest":
    pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y = random_forest(train_X, train_y, val_X, n_estimators=10, min_samples_leaf=5, criterion='gini', print_model_details=False, gridsearch=True)
elif algorithm == "feedforward_neural_network":
    pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y = feedforward_neural_network(train_X, train_y, val_X, hidden_layer_sizes=(100,), max_iter=500, activation='logistic', alpha=0.0001, learning_rate='adaptive', gridsearch=True, print_model_details=False)
else:
    print("done")

#accuracy =  accuracy_score(frame_prob_test_y, test_y)
#print("Accuracy:", accuracy)



