import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Silences unutilized CPU extensions warning thrown when utilizing TensorFlow with a modern CPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sp_data = pd.read_csv('data/all_stocks_5yr.csv')

sp_data['Mid'] = (sp_data[['high','low']].sum(axis=1) / 2.0)
print (sp_data.head)

sp_data = sp_data.drop(columns=['high','low','open','close','volume'])
sp_data_copy = sp_data.copy(deep=True)
print (sp_data.head)

symbols = sp_data['Name'].unique()

for name in symbols:
	tmp = sp_data_copy.loc[sp_data['Name'] == name]
	sp_data_copy = sp_data_copy[tmp.shape[0]:]
	print (sp_data_copy.head(5))
	tmp = tmp.reset_index()
	print (name)
	sp_data[name] = tmp['Mid']


sp_data.to_csv('data/formatted_data.csv')