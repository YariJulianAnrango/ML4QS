import pandas as pd
import numpy as np
from tqdm import tqdm

def Target_classification(df):
    New_weather_steps = df
    New_weather_steps["value_heart"].replace(0, np.nan, inplace=True)

    New_weather_steps["mean_values"] = New_weather_steps.groupby(New_weather_steps.index // (6*24*30))["temp_celsius"].transform("mean")
    New_weather_steps["mean_temp"] = New_weather_steps["temp_celsius"] - New_weather_steps["mean_values"]


    #create combined column
    New_weather_steps["combined"] = np.nan
    for j in tqdm(range(len(New_weather_steps))):
        if New_weather_steps["mean_temp"][j] < (-100) and New_weather_steps["rain"][j] == 0:
            New_weather_steps.loc[j,"combined"] = "ExtraExtraLow0"
        elif New_weather_steps["mean_temp"][j] < (-100) and New_weather_steps["rain"][j] == 1:
            New_weather_steps.loc[j,"combined"] = "ExtraExtraLow1"
        elif New_weather_steps["mean_temp"][j] < (-50) and New_weather_steps["rain"][j] == 0:
            New_weather_steps.loc[j,"combined"] = "ExtraLow0"
        elif New_weather_steps["mean_temp"][j] < (-50) and New_weather_steps["rain"][j] == 1:
            New_weather_steps.loc[j,"combined"] = "ExtraLow1"
        elif New_weather_steps["mean_temp"][j] < (0) and New_weather_steps["rain"][j] == 0:
            New_weather_steps.loc[j,"combined"] = "Low0"
        elif New_weather_steps["mean_temp"][j] < (0) and New_weather_steps["rain"][j] == 1:
            New_weather_steps.loc[j,"combined"] = "Low1"
        elif New_weather_steps["mean_temp"][j] < (50) and New_weather_steps["rain"][j] == 0:
            New_weather_steps.loc[j,"combined"] = "High0"
        elif New_weather_steps["mean_temp"][j] < (50) and New_weather_steps["rain"][j] == 1:
            New_weather_steps.loc[j,"combined"] = "High1"
        elif New_weather_steps["mean_temp"][j] < (100) and New_weather_steps["rain"][j] == 0:
            New_weather_steps.loc[j,"combined"] = "ExtraHigh0"
        elif New_weather_steps["mean_temp"][j] < (100) and New_weather_steps["rain"][j] == 1:
            New_weather_steps.loc[j,"combined"] = "ExtraHigh1"
        elif New_weather_steps["mean_temp"][j] >= (100) and New_weather_steps["rain"][j] == 0:
            New_weather_steps.loc[j,"combined"] = "ExtraExtraHigh0"
        elif New_weather_steps["mean_temp"][j] >= (100) and New_weather_steps["rain"][j] == 1:
            New_weather_steps.loc[j,"combined"] = "ExtraExtraHigh1"
    
    New_weather_steps['start'] = pd.to_datetime(New_weather_steps['start'])
    New_weather_steps = New_weather_steps[~((New_weather_steps['start'].dt.time >= pd.to_datetime('00:00:00').time()) &
          (New_weather_steps['start'].dt.time <= pd.to_datetime('08:00:00').time()))]
    
    #remove na, and columns
    New_weather_steps = New_weather_steps[New_weather_steps['combined'].notna()]
    New_weather_steps = New_weather_steps.drop(columns=['temp_celsius', 'rain', 'mean_temp'])

    return(New_weather_steps)


#df = pd.read_csv("C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\aggregated_data\\merged.csv")
#datacheck = Target_classification(df)
#datacheck.to_csv("datacheck.csv")



