#!/usr/bin/env python
#_*_coding:utf-8_*_

import sys, os, re, platform
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
import checkFasta

def AAINDEX(fastas, **kw):
	if checkFasta.checkFasta(fastas) == False:
		print('Error: for "AAINDEX" encoding, the input fasta sequences should be with equal length. \n\n')
		return 0

	AA = 'ARNDCQEGHILKMFPSTWYV'

	fileAAindex = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + r'\data\AAindex.txt' if platform.system() == 'Windows' else re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + '/data/AAindex.txt'
	with open(fileAAindex) as f:
		records = f.readlines()[1:]

	AAindex = []
	AAindexName = []
	for i in records:
		AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
		AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)

	index = {}
	for i in range(len(AA)):
		index[AA[i]] = i

	encodings = []
	header = ['#']
	for pos in range(1, len(fastas[0][1]) + 1):
		for idName in AAindexName:
			header.append('SeqPos.' + str(pos) + '.' + idName)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], i[1]
		code = [name]
		for aa in sequence:
			if aa == '-':
				for j in AAindex:
					code.append(0)
				continue
			for j in AAindex:
				code.append(j[index[aa]])
		encodings.append(code)

	return encodings
# import pandas as pd
# import numpy as np
# def feature_generator(file_path, temp_file_path):
#     # pd.set_option('display.max_colwidth', 500)
#     # pd.set_option('display.max_columns', 1000)
#     # pd.set_option('display.width', 1000)
#     f = open(file_path, 'r', encoding='utf-8')
#     fasta_list = np.array(f.readlines())
#     aa_feature_list = []
#     # print(fasta_list)len(fasta_list)
#     for flag in range(0, len(fasta_list), 2):
#         fasta_str = [[fasta_list[flag].strip('\n').strip(), fasta_list[flag + 1].strip('\n').strip()]]
#
#         aac_output = AAINDEX(fasta_str)
#         aac_output[1].remove(aac_output[1][0])
#         # print(dpc_output)
#         # dpc_feature = []
#         aac_feature = aac_output[1][:]
#         # print(dpc_feature)
#         aa_feature_list.append(aac_feature)
#     aa_feature_list = pd.DataFrame(aa_feature_list)
#     aa_feature_list = aa_feature_list.iloc[:, :]
#     coloumnname = []
#     for i in range(len(aa_feature_list.columns)):
#         x = 'AA'+str(i+1)
#         coloumnname.append(x)
#     # print(coloumnname)
#     aa_feature_list.columns = coloumnname
#     print(aa_feature_list.shape)
#     # return aa_feature_list.to_csv(temp_file_path, sep=',')
#
#
#
# if __name__ == '__main__':
#     feature_generator('/Users/ggcl7/Desktop/AIP-dataset/ML_dataset/test-all.txt', '/Users/ggcl7/Desktop/AIP-dataset/ML_dataset/test-allaaindex.csv')