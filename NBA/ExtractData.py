import pandas as pd

df = pd.read_csv('Data/All_Game_2015-16.csv')
print(type(df['teamVector'][0]))