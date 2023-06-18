from functions.combine_data_ import create_dataset
from functions.Target_transformation.target_transformation import Target_classification
from functions.Missing_data.missing_values import interpolate
from functions.Feature_engineering.feature_engineering import Feature_Engineering



print('This will take a while...')
#combine_data_
dat = create_dataset('/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/data_used/weather_2022_2023.txt', delta_t=10)
dat.merge_data_weather(dir_steps='/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/data_used/StepCount.csv',
                        dir_heart='/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/data_used/HeartRate.csv',
                        dir_workout='/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/data_used/Workout.csv',
                        start_date='2022-01-01 00:00:00',
                        end_date='2023-06-06 00:00:00')

# Target classification
print('target classification')
weather_steps = Target_classification(dat.data_frame)

print('interpolation')
#  Interpolation of na's in heartrate
weather_steps = interpolate(weather_steps)

# Feature engineering
feat_eng = Feature_Engineering(weather_steps)

print('rolling_functions')
feat_eng.rolling_functions(100)

print('fourier_transformation')
feat_eng.fourier_transformation()
feat_eng.one_hot_encoding()
#
# # Feature selection
selected_features, ordered_features, ordered_scores, train_X, test_X, train_y, test_y, val_X, val_y = feat_eng.feature_selection(20)

feat_eng.data_frame[selected_features].to_csv('features_selected.csv')





