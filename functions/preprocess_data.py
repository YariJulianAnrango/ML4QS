import pandas as pd

from functions.combine_data_ import create_dataset
from functions.Target_transformation.target_transformation import Target_classification
from functions.Missing_data.missing_values import interpolate
from functions.Feature_engineering.feature_engineering import Feature_Engineering
from functions.train_test_split import temporal_split
from functions.Feature_engineering.feature_engineering import normalise_dataset
import time
from functions.Feature_engineering.feature_engineering import one_hot_encoding


start = time.time()
print('This will take a while...')
#combine_data_
# dat = create_dataset('/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/data_used/weather_2022_2023.txt', delta_t=10)
# dat.merge_data_weather(dir_steps='/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/data_used/StepCount.csv',
#                         dir_heart='/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/data_used/HeartRate.csv',
#                         dir_workout='/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/data_used/Workout.csv',
#                         start_date='2022-01-01 00:00:00',
#                         end_date='2023-06-06 00:00:00')

dat = pd.read_csv('/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/aggregated_data/merged.csv', index_col = 0)

# Target classification
print('target classification')
weather_steps = Target_classification(dat)

print('interpolation')
#  Interpolation of na's in heartrate
weather_steps = interpolate(weather_steps)

weather_steps = one_hot_encoding(weather_steps)

weather_steps = normalise_dataset(weather_steps
                                  )
# Train/test/val split
train, test, val = temporal_split(weather_steps)

# Feature engineering
print('train set fe')
feat_eng = Feature_Engineering(train)
feat_eng.rolling_functions(100)
feat_eng.fourier_transformation()

# Feature engineering for validation and test set
print('val set fe')
feat_val = Feature_Engineering(val)
feat_val.rolling_functions(100)
feat_val.fourier_transformation()

print('test set fe')
feat_test = Feature_Engineering(test)
feat_test.rolling_functions(100)
feat_test.fourier_transformation()

# # Feature selection
print('feature selection')
selected_features, ordered_features, ordered_scores = feat_eng.feature_selection(20, feat_val.data_frame)

selected_features.append('combined')

feat_eng.data_frame[selected_features].dropna().to_csv('train.csv')
feat_val.data_frame[selected_features].dropna().to_csv('val.csv')
feat_test.data_frame[selected_features].dropna().to_csv('test.csv')


end = time.time()

total_time = end - start
print("\n"+ str(total_time))




