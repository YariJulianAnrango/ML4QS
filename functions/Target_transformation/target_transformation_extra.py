import pandas as pd
import numpy as np
from tqdm import tqdm

#Het begint bij merged.csv. Deze is in combine_data.py aangemaakt. De data is hierin verkort naar bruikbare variabelen
#Deze dataset wordt toegepast in deze Target_classification functie
#De functie maakt aan het einde de dataset aan genaamd 'datacheck.csv'. Deze wordt gebruikt in EDA.ipynb



def Target_classification(df):
    New_weather_steps = df
    #Alle zero values van de kolom Value_heart worden omgezet naar None
    New_weather_steps["value_heart"].replace(0, np.nan, inplace=True)

    #Een nieuwe kolom 'mean_values' wordt aangemaakt waarin per maand een gemiddelde temperatuur staat (om de 10 minuten is er een meting
    # dus 6 metingen per uur. 24 uur per dag. 30 dagen in de maand. Dat is 6*24*30 metingen waar een gemiddelde van wordt bepaald)
    New_weather_steps["mean_values"] = New_weather_steps.groupby(New_weather_steps.index // (6*24*30))["temp_celsius"].transform("mean")
    
    #Vervolgens worden de werkelijke temperatuur waardes afgetrokken van het gemiddelde temperatuur van die maand. Dit 
    #zorgt er namelijk voor dat een temperatuur relatief is. 20 graden is in februari super warm, maar in juli is dat niet zo warm.
    #Deze waardes worden in de kolom 'mean_temp' gezet.
    New_weather_steps["mean_temp"] = New_weather_steps["temp_celsius"] - New_weather_steps["mean_values"]


    #Aan de hand van 'mean_temp' en 'rain' wordt bepaald wat de weerconditie is. Wanneer het relatief zeer koud 
    #is en het regent, dan wordt het element in kolom 'combined' toegewezen aan de waarde 'ExtraExtraLow1'
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
        elif New_weather_steps["mean_temp"][j] < (-10) and New_weather_steps["rain"][j] == 0:
            New_weather_steps.loc[j,"combined"] = "Low0"
        elif New_weather_steps["mean_temp"][j] < (-10) and New_weather_steps["rain"][j] == 1:
            New_weather_steps.loc[j,"combined"] = "Low1"
        elif New_weather_steps["mean_temp"][j] < (10) and New_weather_steps["rain"][j] == 0:
            New_weather_steps.loc[j,"combined"] = "Normal0"
        elif New_weather_steps["mean_temp"][j] < (10) and New_weather_steps["rain"][j] == 1:
            New_weather_steps.loc[j,"combined"] = "Normal1"
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
    
    #New_weather_steps['start'] = pd.to_datetime(New_weather_steps['start'])
    #New_weather_steps = New_weather_steps[~((New_weather_steps['start'].dt.time >= pd.to_datetime('00:00:00').time()) &
    #      (New_weather_steps['start'].dt.time <= pd.to_datetime('08:00:00').time()))]
    
    #remove na, and columns
    #New_weather_steps = New_weather_steps[New_weather_steps['combined'].notna()]
    New_weather_steps = New_weather_steps.drop(columns=['temp_celsius', 'rain', 'mean_temp', 'mean_values'])

    return(New_weather_steps)


#De datacheck.csv wordt aangemaakt zodat hier vervolgens EDA op gedaan kan worden. Deze hoef je maar 1x te runnen.
#Momenteel staat hij al opgeslagen, dus heb ik hem gecommend.
#df = pd.read_csv("C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\aggregated_data\\merged.csv")
#dff = Target_classification(df)
#dff.to_csv('datacheck.csv')

