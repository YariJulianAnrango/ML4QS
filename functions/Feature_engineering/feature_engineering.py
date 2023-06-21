import pandas as pd
from Python3Code.Chapter4.FrequencyAbstraction import FourierTransformation
from Python3Code.Chapter7.FeatureSelection import FeatureSelectionClassification
from sklearn.preprocessing import OneHotEncoder


class Feature_Engineering:
    def __init__(self, df):
        self.data_frame = df

    def rolling_functions(self, max_window_size):
        # Rolling mean
        for l in range(1, max_window_size + 1):
            rolling_mean = self.data_frame[['value_steps', 'value_heart']].rolling(window=l, min_periods=1).mean()
            rolling_mean.columns = ['steps_rmean_l' + str(l), 'heart_rmean_l' + str(l)]
            self.data_frame = pd.concat([self.data_frame, rolling_mean], axis=1)

        # Rolling max
        for l in range(1, max_window_size + 1):
            rolling_max = self.data_frame[['value_steps', 'value_heart']].rolling(window=l, min_periods=1).max()
            rolling_max.columns = ['steps_rmax_l' + str(l), 'heart_rmax_l' + str(l)]
            self.data_frame = pd.concat([self.data_frame, rolling_max], axis=1)

        # Rolling min
        for l in range(1, max_window_size + 1):
            rolling_min = self.data_frame[['value_steps', 'value_heart']].rolling(window=l, min_periods=1).min()
            rolling_min.columns = ['steps_rmin_l' + str(l), 'heart_rmin_l' + str(l)]
            self.data_frame = pd.concat([self.data_frame, rolling_min], axis=1)

        # Rolling std
        for l in range(2, max_window_size + 1):
            rolling_std = self.data_frame[['value_steps', 'value_heart']].rolling(window=l, min_periods=1).std()
            rolling_std.columns = ['steps_rstd_l' + str(l), 'heart_rstd_l' + str(l)]
            self.data_frame = pd.concat([self.data_frame, rolling_std], axis=1)

        # Rolling sum
        for l in range(1, max_window_size + 1):
            rolling_sum = self.data_frame[['value_steps', 'value_heart']].rolling(window=l, min_periods=1).sum()
            rolling_sum.columns = ['steps_rsum_l' + str(l), 'heart_rsum_l' + str(l)]
            self.data_frame = pd.concat([self.data_frame, rolling_sum], axis=1)

    def fourier_transformation(self, cols = ['value_heart', 'value_steps']):
        FreqAbs = FourierTransformation()
        milliseconds_per_instance = (pd.to_datetime(self.data_frame.loc[1, 'start']) - pd.to_datetime(
            self.data_frame.loc[0, 'start'])).seconds * 1000

        fs = float(1000000) / milliseconds_per_instance
        ws = int(float(100000000) / milliseconds_per_instance)

        for c in cols:
            self.data_frame = FreqAbs.abstract_frequency(self.data_frame, [c], ws, fs)

        self.data_frame = self.data_frame.drop(['start','end'], axis = 1)

    def one_hot_encoding(self, cols = 'value_workout'):
        dataset = self.data_frame.drop(['start', 'end'], axis=1)
        enc = OneHotEncoder()
        transformed_mat = enc.fit_transform(dataset[[cols]])
        transformed = pd.DataFrame.sparse.from_spmatrix(transformed_mat)
        transformed.columns = enc.categories_[0]

        dataset = pd.concat([dataset, transformed], axis=1)
        dataset = dataset.drop(cols, axis=1)
        dataset[enc.categories_[0]] = dataset[enc.categories_[0]].sparse.to_dense()

        self.data_frame = dataset

    def feature_selection(self, max_features, val):
        train = self.data_frame.dropna()
        train = train.drop(['value_steps_max_freq', 'value_steps_freq_weighted'], axis = 1)
        train_X = train.loc[:,train.columns != 'combined']
        train_y = train[['combined']]

        print('val',val)
        val = val.drop(['value_steps_max_freq', 'value_steps_freq_weighted'], axis = 1)
        val = val.dropna()
        val_X = val.loc[:,val.columns != 'combined']
        val_y = val[['combined']]

        print('train_x',train_X)
        print('train_y',train_y)
        print('val_x',val_X)
        print('val_y',val_y)
        feat_sel = FeatureSelectionClassification()
        selected_features, ordered_features, ordered_scores = feat_sel.forward_selection(max_features, train_X, val_X, train_y, val_y)

        return selected_features, ordered_features, ordered_scores

def one_hot_encoding(df, cols = 'value_workout'):
    enc = OneHotEncoder()
    transformed_mat = enc.fit_transform(df[[cols]])
    transformed = pd.DataFrame.sparse.from_spmatrix(transformed_mat)
    transformed.columns = enc.categories_[0]

    dataset = pd.concat([df, transformed], axis=1)
    dataset = dataset.drop(cols, axis=1)
    dataset[enc.categories_[0]] = dataset[enc.categories_[0]].sparse.to_dense()

    return dataset


