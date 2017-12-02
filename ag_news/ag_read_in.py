import sys 
import pandas as pd
'''
in total 4 categories: World(1) Sports(2) Business(3) Sci/Tech(4)
return a pandas dataframe
'''

def read_in(option):
	file_string=''
	if option =='train':
		file_string='train.csv'
	elif option == 'test':
		file_string='test.csv'

	column_names= ['class','title','description'] 
	df = pd.read_csv(file_string, header = None, names = column_names)
	print df.head(10)
	return df


read_in('test')


