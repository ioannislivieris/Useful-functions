# utility script to convert csv files to parquet 
import os
import numpy as np
import pandas as pd
import sys
import argparse
# import modin.pandas as md
# os.environ["MODIN_ENGINE"]="ray"

# usage:   convert_to_parquet.py --p <path> --f <filename without ext> --v <show info> 
# example: convert_to_parquet.py --p './' --f 'FAN TEMPERATURE' --v True

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Convert csv file to parquet!") 
	# "usage:   convert_to_parquet.py --p <path> --f <filename> --v <show info>" \
	# "example: convert_to_parquet.py --p './' --f 'FAN TEMPERATURE' --v True")

	# parser.print_help()
	
	parser.add_argument("--p", default='./', type=str, help='file path (default: current directory)')
	parser.add_argument("--f", default=None, type=str, required=True, help='filename to convert without csv ext')
	parser.add_argument("--v", default=False, help='show info (default: False)')

	args = parser.parse_args()
	filename = args.f
	path = args.p
	verbose = args.v

	print('> Load df')
	df = pd.read_csv(path+filename+'.csv', index_col=[0])    # , index=True

	if verbose: 
		print('Convert to parquet the file:', path+filename+'.csv')
		print('Load df with shape:', df.shape)
		print('Columns:', df.columns)
		print('Start/End date:', df.index[-1], '--', df.index[0])

	df.to_parquet(path+filename+'.pq')
	print('> df converted to parquet!')

# -------------------------------------
