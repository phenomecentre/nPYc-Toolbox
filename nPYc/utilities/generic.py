"""
Generic Utility functions
"""
import json

def removeDuplicateColumns(df):
	"""
	Removes duplicate columns from the passed dataframe
	Looks for columns that end with _x or _y
	"""
	cols = [c for c in df.columns if c[-2:] != '_y']
	df = df[cols]
	df = df.rename(columns=lambda x: x if x[-2:] != '_x' else x.replace('_x', ''))
	return df


def removeTrailingColumnNumbering(column_list):
	"""
	When pandas finds columns with same name, it numbers them
	This function receives a list of column names and removes the numbering if found
	Looks for columns that end with .1, .2, .3 and so on
	"""
	import re
	tmp = []
	for s in column_list:
		x = re.search('\.{1}\d+',s)
		if x != None:
			i = x.span()[0] #index of the .
			tmp.append(s[:i])
		else:
			tmp.append(s)

	return tmp

def print_dict(message, dict_obj):  
    """
    Use JSON to print out a dict or list of dicts
    in an easily readable format
    """
    my_complex_dict = json.dumps(dict_obj, indent=4, default=str, allow_nan=True)
    print(f"{message}: {my_complex_dict}")  
 	
	