import pandas as pd
import numpy as np

def Target_classification(Weather_steps_df):
    #load dataset
    New_weather_steps = pd.read_csv(Weather_steps_df)
    New_weather_steps = New_weather_steps.drop("Unnamed: 0",axis=1)
    New_weather_steps["value_heart"].replace(0, np.nan, inplace=True)

    #categorize weather
    for i in range(len(New_weather_steps)):
        if New_weather_steps["temp_celsius"][i] < (0):
            New_weather_steps["temp_celsius"][i] = "Freezing"
        elif New_weather_steps["temp_celsius"][i] < (10*10):
            New_weather_steps["temp_celsius"][i] = "Cold"
        elif New_weather_steps["temp_celsius"][i] < (15*10):
            New_weather_steps["temp_celsius"][i] = "Chilly"
        elif New_weather_steps["temp_celsius"][i] < (20*10):
            New_weather_steps["temp_celsius"][i] = "Comfortable"
        elif New_weather_steps["temp_celsius"][i] < (25*10):
            New_weather_steps["temp_celsius"][i] = "Warm"
        elif New_weather_steps["temp_celsius"][i] >= (25*10):
            New_weather_steps["temp_celsius"][i] = "Hot"
    
    #create combined column
    New_weather_steps["combined"] = np.nan
    for j in range(len(New_weather_steps)):
        if New_weather_steps["temp_celsius"][j] == "Freezing" and New_weather_steps["rain"][j] == 0:
            New_weather_steps["combined"][j] = 1
        elif New_weather_steps["temp_celsius"][j] == "Freezing" and New_weather_steps["rain"][j] == 1:
            New_weather_steps["combined"][j] = 2
        elif New_weather_steps["temp_celsius"][j] == "Cold" and New_weather_steps["rain"][j] == 0:
            New_weather_steps["combined"][j] = 3
        elif New_weather_steps["temp_celsius"][j] == "Cold" and New_weather_steps["rain"][j] == 1:
            New_weather_steps["combined"][j] = 4
        elif New_weather_steps["temp_celsius"][j] == "Chilly" and New_weather_steps["rain"][j] == 0:
            New_weather_steps["combined"][j] = 5
        elif New_weather_steps["temp_celsius"][j] == "Chilly" and New_weather_steps["rain"][j] == 1:
            New_weather_steps["combined"][j] = 6
        elif New_weather_steps["temp_celsius"][j] == "Comfortable" and New_weather_steps["rain"][j] == 0:
            New_weather_steps["combined"][j] = 7
        elif New_weather_steps["temp_celsius"][j] == "Comfortable" and New_weather_steps["rain"][j] == 1:
            New_weather_steps["combined"][j] = 8
        elif New_weather_steps["temp_celsius"][j] == "Warm" and New_weather_steps["rain"][j] == 0:
            New_weather_steps["combined"][j] = 9
        elif New_weather_steps["temp_celsius"][j] == "Warm" and New_weather_steps["rain"][j] == 1:
            New_weather_steps["combined"][j] = 10
        elif New_weather_steps["temp_celsius"][j] == "Hot" and New_weather_steps["rain"][j] == 0:
            New_weather_steps["combined"][j] = 11
        elif New_weather_steps["temp_celsius"][j] == "Hot" and New_weather_steps["rain"][j] == 1:
            New_weather_steps["combined"][j] = 12
    
    #remove na, and columns
    New_weather_steps = New_weather_steps[New_weather_steps['combined'].notna()]
    New_weather_steps = New_weather_steps.drop(columns=['temp_celsius', 'rain'])

    for g in range(len(New_weather_steps)):
        if New_weather_steps["value_workout"][g] == 'no workout':
            New_weather_steps["value_workout"][g] = 1
        elif New_weather_steps["value_workout"][g] == 'HKWorkoutActivityTypeWalking':
            New_weather_steps["value_workout"][g] = 2
        elif New_weather_steps["value_workout"][g] == 'HKWorkoutActivityTypeRunning':
            New_weather_steps["value_workout"][g] = 3
        elif New_weather_steps["value_workout"][g] == 'HKWorkoutActivityTypeCooldown':
            New_weather_steps["value_workout"][g] = 4
        elif New_weather_steps["value_workout"][g] == 'HKWorkoutActivityTypeTraditionalStrengthTraining':
            New_weather_steps["value_workout"][g] = 5
        elif New_weather_steps["value_workout"][g] == 'HKWorkoutActivityTypeSkatingSports':
            New_weather_steps["value_workout"][g] = 6
        elif New_weather_steps["value_workout"][g] == 'HKWorkoutActivityTypeClimbing':
            New_weather_steps["value_workout"][g] = 7
        elif New_weather_steps["value_workout"][g] == 'HKWorkoutActivityTypeCycling':
            New_weather_steps["value_workout"][g] = 8
        elif New_weather_steps["value_workout"][g] == 'HKWorkoutActivityTypeFunctionalStrengthTraining':
            New_weather_steps["value_workout"][g] = 9
    return(New_weather_steps)

Target_weather_steps = Target_classification("C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\New_weather_steps.csv")
Target_weather_steps.to_csv('Target_weather_steps.csv')


