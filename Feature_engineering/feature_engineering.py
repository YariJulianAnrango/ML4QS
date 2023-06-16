import pandas as pd
from Python3Code.Chapter4.FrequencyAbstraction import FourierTransformation
from Rolling_functions import rolling_functions
from Python3Code.Chapter7.FeatureSelection import FeatureSelectionRegression
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv('/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/merged.csv', index_col=0)

# Rolling functions
print('rolling functions')
# dataset = rolling_functions(dataset, 100)

# Fourier Transformation
print('FourierTransformation')
FreqAbs = FourierTransformation()
milliseconds_per_instance = (pd.to_datetime(dataset.loc[1,'start']) - pd.to_datetime(dataset.loc[0,'start'])).seconds*1000

fs = float(1000000)/milliseconds_per_instance
ws = int(float(100000000)/milliseconds_per_instance)

dataset = FreqAbs.abstract_frequency(dataset, ['value_heart'], ws, fs)
dataset = FreqAbs.abstract_frequency(dataset, ['value_steps'], ws, fs)

# One hot encoding
# dataset = dataset.drop(['start','end'], axis = 1)
# enc = OneHotEncoder()
# transformed_mat = enc.fit_transform(dataset[['value_workout']])
# transformed = pd.DataFrame.sparse.from_spmatrix(transformed_mat)
# transformed.columns = enc.categories_[0]
#
# dataset = pd.concat([dataset, transformed], axis = 1)
# dataset = dataset.drop('value_workout', axis = 1)
# dataset[enc.categories_[0]] = dataset[enc.categories_[0]].sparse.to_dense()


# #Feature selection
# dataset = dataset.drop(['value_steps_max_freq', 'value_steps_freq_weighted','start','end'], axis = 1)
# dataset = dataset.dropna()
# print('Feature selection')
# feat_sel = FeatureSelectionRegression()
# X_train = dataset.loc[:,dataset.columns.drop(['combined'])]
# y_train = dataset.loc[:,['combined']]
# selected_features, ordered_features, ordered_scores = feat_sel.forward_selection(100, X_train, y_train)

