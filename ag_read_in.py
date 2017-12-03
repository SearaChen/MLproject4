import sys 
import pandas as pd
import numpy as np
'''
in total 4 categories: World(1) Sports(2) Business(3) Sci/Tech(4)
takes in argument whether the test/train desired, to keep in consistant with original test train split configuation 
return a pandas dataframe
'''

def read_in(option):
	file_string=''
	if option =='train':
		file_string='ag_news_csv/train.csv'
	elif option == 'test':
		file_string='ag_news_csv/test.csv'

	column_names= ['class','title','description'] 
	df = pd.read_csv(file_string, header = None, names = column_names)
	df = np.asarray(df)
	return df





