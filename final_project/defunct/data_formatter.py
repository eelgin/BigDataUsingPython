import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

directory = 'data/individual_stocks_5yr'
files = os.listdir(directory)

sp_data = pd.DataFrame(columns=['date'])

for file in files:

	tmp = pd.read_csv(directory+'/'+file)

	name = file.rpartition('_')[0]

	print (name)

	tmp[name] = (tmp[['high','low']].sum(axis=1) / 2.0)
	tmp =  tmp.drop(columns=['high','low','open','close','volume','Name'])

	#tmp[]

	sp_data = pd.merge(sp_data, tmp, how='outer', on='date')

	print (sp_data.head(5))

	del tmp

indx_data = sp_data[list(sp_data.columns)].sum(axis=1)

sp_data.insert(loc=1, column='S&P500', value=indx_data)
print (sp_data.head(5))

sp_data.to_csv('data/formatted_data.csv')
