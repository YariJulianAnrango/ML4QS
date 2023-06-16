from combine_data_ import create_dataset
from target_transformation import Target_classification
from missing_values import interpolate
from missing_values import kalman_filter
from Split_and_models import Stratified_Split
from Split_and_models import kfold_crossvalidation
from Split_and_models import k_nearest_neighbor
from Split_and_models import support_vector_machine_without_kernel
from Split_and_models import naive_bayes
from Split_and_models import random_forest
from Split_and_models import feedforward_neural_network

print("packages loaded")

#Choose the method and algorithm
method = "interpolate"
algorithm = "k_nearest_neighbor"

#combine_data_
data_table = create_dataset('C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\weather_2022_2023.txt', delta_t=10)
weather_steps = data_table.merge_data_weather(dir_steps='C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\StepCount.csv',
                        dir_heart='C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\HeartRate.csv',
                        dir_workout='C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\Workout.csv',
                        start_date='2022-01-01 00:00:00',
                        end_date='2023-06-06 00:00:00')

#weather_steps = create_dataset.data_table
weather_steps.to_csv('weather_steps.csv')

#Target_transformation
weather_steps = Target_classification('C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\weather_steps.csv')
weather_steps.to_csv('weather_steps.csv')

#missing_values
if method == 'interpolate':
    weather_steps = interpolate('C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\weather_steps.csv')
elif method == 'kalman_filter':
    weather_steps = kalman_filter('C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\weather_steps.csv')
weather_steps.to_csv('weather_steps.csv')

#train/test split
split_method = "kfold_crossvalidation"
if split_method == "Stratified_Split":
     train_X, test_X, train_y, test_y, val_X, val_y = Stratified_Split('C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\weather_steps.csv')
elif split_method == "kfold_crossvalidation":
     train_X, test_X, train_y, test_y, val_X, val_y = kfold_crossvalidation('C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\weather_steps.csv')
else:
    print("finished")

#Model
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
    print("finished")

