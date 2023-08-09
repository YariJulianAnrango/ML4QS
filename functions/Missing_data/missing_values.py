import pandas as pd
import numpy as np
from pykalman import KalmanFilter

#In EDA.ipynb zagen we dat 'value_heart' van merged.csv veel Nan heeft. Deze Nan willen we vervangen voor realistische waardes
#Om zo'n inputation te doen, kijken we naar interpolate and kalman_filter. 

#interpolate. Stel je hebt: 7,Nan,8, Dan maakt de interpolate zelf van de Nan een 7,5
def interpolate(New_weather_steps):
    New_weather_steps['value_heart'] = New_weather_steps['value_heart'].interpolate()
    return(New_weather_steps)


#
def kalman_filter(New_weather_steps):
    New_weather_steps = pd.read_csv(New_weather_steps)
    New_weather_steps = New_weather_steps.drop("Unnamed: 0",axis=1)

    # Initialize the Kalman filter with the trivial transition and observation matrices.
    kf = KalmanFilter(transition_matrices=[[1]], observation_matrices=[[1]])

    numpy_array_state = New_weather_steps["value_heart"].values
    numpy_array_state = numpy_array_state.astype(np.float32)
    numpy_matrix_state_with_mask = np.ma.masked_invalid(numpy_array_state)

    # Find the best other parameters based on the data (e.g. Q)
    kf = kf.em(numpy_matrix_state_with_mask, n_iter=5)

    # And apply the filter.
    (new_data, filtered_state_covariances) = kf.filter(numpy_matrix_state_with_mask)

    New_weather_steps["value_heart" + '_kalman'] = new_data
    return(New_weather_steps)


#check of de functie van interpolate het doet (duurt nog geen 1 sec bij mij)
df = pd.read_csv("C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\aggregated_data\\merged.csv")
dff = interpolate(df)
print(dff['value_heart'])
