import pandas as pd
from Python3Code.Chapter4.TemporalAbstraction import CategoricalAbstraction
from Python3Code.Chapter4.FrequencyAbstraction import FourierTransformation
from Rolling_functions import rolling_functions
from Python3Code.util.VisualizeDataset import VisualizeDataset

dataset = pd.read_csv('/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/merged.csv', index_col=0)

# Rolling functions
dataset = rolling_functions(dataset, 5)

# Categorical Abstraction
# dataset['value_workout'] = str(dataset['value_workout'])
# CatAbs = CategoricalAbstraction()
# dataset = CatAbs.abstract_categorical(dataset, ['value_workout'], ['exact'], 0.03, 5, 2)

# Fourier Transformation
FreqAbs = FourierTransformation()
milliseconds_per_instance = (pd.to_datetime(dataset.loc[1,'start']) - pd.to_datetime(dataset.loc[0,'start'])).seconds*1000

fs = float(1000000)/milliseconds_per_instance
ws = int(float(100000000)/milliseconds_per_instance)

dataset = FreqAbs.abstract_frequency(dataset, ['value_heart'], ws, fs)
dataset = FreqAbs.abstract_frequency(dataset, ['value_steps'], ws, fs)


