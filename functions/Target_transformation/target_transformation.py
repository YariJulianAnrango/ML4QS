import pandas as pd
import numpy as np
from tqdm import tqdm

def Target_classification(df):
    New_weather_steps = df
    print(New_weather_steps)
    New_weather_steps["value_heart"].replace(0, np.nan, inplace=True)

    # #categorize weather
    # New_weather_steps['temp_class'] = 'unclassed'
    # for i in range(len(New_weather_steps)):
    #     if New_weather_steps.loc[i,"temp_celsius"] < (0):
    #         New_weather_steps.loc[i,"temp_class"] = "Freezing"
    #     elif New_weather_steps.loc[i,"temp_celsius"] < (10*10):
    #         New_weather_steps.loc[i,"temp_class"] = "Cold"
    #     elif New_weather_steps.loc[i,"temp_celsius"] < (15*10):
    #         New_weather_steps.loc[i,"temp_class"] = "Chilly"
    #     elif New_weather_steps.loc[i,"temp_celsius"] < (20*10):
    #         New_weather_steps.loc[i,"temp_class"] = "Comfortable"
    #     elif New_weather_steps.loc[i,"temp_celsius"] < (25*10):
    #         New_weather_steps.loc[i,"temp_class"] = "Warm"
    #     elif New_weather_steps.loc[i,"temp_celsius"] >= (25*10):
    #         New_weather_steps.loc[i,"temp_class"] = "Hot"
    
    #create combined column
    New_weather_steps["combined"] = np.nan
    for j in tqdm(range(len(New_weather_steps))):
        if New_weather_steps.loc[j,"temp_celsius"] < (0) and New_weather_steps["rain"][j] == 0:
            New_weather_steps.loc[j,"combined"] = "Freezing0"
        elif New_weather_steps.loc[j,"temp_celsius"] < (0) and New_weather_steps["rain"][j] == 1:
            New_weather_steps.loc[j,"combined"] = "Freezing1"
        elif New_weather_steps.loc[j,"temp_celsius"] < (10*10) and New_weather_steps["rain"][j] == 0:
            New_weather_steps.loc[j,"combined"] = "Cold0"
        elif New_weather_steps.loc[j,"temp_celsius"] < (10*10) and New_weather_steps["rain"][j] == 1:
            New_weather_steps.loc[j,"combined"] = "Cold1"
        elif New_weather_steps.loc[j,"temp_celsius"] < (15*10) and New_weather_steps["rain"][j] == 0:
            New_weather_steps.loc[j,"combined"] = "Chilly0"
        elif New_weather_steps.loc[j,"temp_celsius"] < (15*10) and New_weather_steps["rain"][j] == 1:
            New_weather_steps.loc[j,"combined"] = "Chilly1"
        elif New_weather_steps.loc[j,"temp_celsius"] < (20*10) and New_weather_steps["rain"][j] == 0:
            New_weather_steps.loc[j,"combined"] = "Comfortable0"
        elif New_weather_steps.loc[j,"temp_celsius"] < (20*10) and New_weather_steps["rain"][j] == 1:
            New_weather_steps.loc[j,"combined"] = "Comfortable1"
        elif New_weather_steps.loc[j,"temp_celsius"] < (25*10) and New_weather_steps["rain"][j] == 0:
            New_weather_steps.loc[j,"combined"] = "Warm0"
        elif New_weather_steps.loc[j,"temp_celsius"] < (25*10) and New_weather_steps["rain"][j] == 1:
            New_weather_steps.loc[j,"combined"] = "Warm1"
        elif New_weather_steps.loc[j,"temp_celsius"] >= (25*10) and New_weather_steps["rain"][j] == 0:
            New_weather_steps.loc[j,"combined"] = "Hot0"
        elif New_weather_steps.loc[j,"temp_celsius"] >= (25*10) and New_weather_steps["rain"][j] == 1:
            New_weather_steps.loc[j,"combined"] = "Hot1"
    
    #remove na, and columns
    New_weather_steps = New_weather_steps[New_weather_steps['combined'].notna()]
    New_weather_steps = New_weather_steps.drop(columns=['temp_celsius', 'rain'])

    return(New_weather_steps)


df = pd.read_csv("C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\aggregated_data\\merged.csv")
datacheck = Target_classification(df)
datacheck.to_csv("datacheck.csv")



