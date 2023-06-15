import pandas as pd

def rolling_functions(df, max_window_size):

    # Rolling mean
    for l in range(1,max_window_size + 1):
        rolling_mean = df[['value_steps','value_heart']].rolling(window=l, min_periods=1).mean()
        rolling_mean.columns = ['steps_rmean_l'+str(l), 'heart_rmean_l'+str(l)]
        df = pd.concat([df, rolling_mean], axis = 1)

    # Rolling max
    for l in range(1, max_window_size + 1):
        rolling_max = df[['value_steps', 'value_heart']].rolling(window=l, min_periods=1).max()
        rolling_max.columns = ['steps_rmax_l' + str(l), 'heart_rmax_l' + str(l)]
        df = pd.concat([df, rolling_max], axis=1)

    # Rolling min
    for l in range(1, max_window_size + 1):
        rolling_min = df[['value_steps', 'value_heart']].rolling(window=l, min_periods=1).min()
        rolling_min.columns = ['steps_rmin_l' + str(l), 'heart_rmin_l' + str(l)]
        df = pd.concat([df, rolling_min], axis=1)

    # Rolling std
    for l in range(2, max_window_size + 1):
        rolling_std = df[['value_steps', 'value_heart']].rolling(window=l, min_periods=1).std()
        rolling_std.columns = ['steps_rstd_l' + str(l), 'heart_rstd_l' + str(l)]
        df = pd.concat([df, rolling_std], axis=1)

    # Rolling sum
    for l in range(1, max_window_size + 1):
        rolling_sum = df[['value_steps', 'value_heart']].rolling(window=l, min_periods=1).sum()
        rolling_sum.columns = ['steps_rsum_l' + str(l), 'heart_rsum_l' + str(l)]
        df = pd.concat([df, rolling_sum], axis=1)

    return df

dat = pd.read_csv('/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/merged.csv')
x = rolling_functions(dat, 5)