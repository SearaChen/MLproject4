import sys 
import pandas as pd
import numpy as np
'''
in total 4 categories: World(1) Sports(2) Business(3) Sci/Tech(4)
takes in argument whether the test/train desired, to keep in consistant with original test train split configuation 
return a pandas dataframe
'''

def read_in(folder,option):
	file_string=''
	if option =='train':
		file_string= folder +'/train.csv'
	elif option == 'test':
		file_string= folder +'/test.csv'

	column_names= ['class','title','description'] 
	df = pd.read_csv(file_string, header = None, names = column_names)
	df = np.asarray(df)
	return df





