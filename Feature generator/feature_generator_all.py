import numpy as np
from BINARY import *
from AAINDEX import *
from BLOSUM62 import *
from ZSCALE import *
# , temp_file_path
from torch.utils.data import Dataset
import numpy as np
def feature_generator(file_path, temp_file_path):
    f = open(file_path, 'r', encoding='utf-8')
    fasta_list = np.array(f.readlines())
    aa_feature_list = []
    for flag in range(0, len(fasta_list), 2):
        fasta_str = [[fasta_list[flag].strip('\n').strip(), fasta_list[flag + 1].strip('\n').strip()]]
        bin_output = BINARY(fasta_str)
        aai_output = AAINDEX(fasta_str)
        blo_output = BLOSUM62(fasta_str)
        zsl_output = ZSCALE(fasta_str)
        if 'Positive' in bin_output[1][0].split('>')[1]:
            feature_id = 1
        else:
            feature_id = 0
        bin_output[1].remove(bin_output[1][0])
        aai_output[1].remove(aai_output[1][0])
        blo_output[1].remove(blo_output[1][0])
        zsl_output[1].remove(zsl_output[1][0])
        bin_feature = []
        aai_feature = []
        blo_feature = []
        zsl_feature = []
        for i in range(0, len(bin_output[1]), 20):
            temp = bin_output[1][i:i + 20]
            bin_feature.append(temp)
        for i in range(0, len(aai_output[1]), 531):
            temp = [float(i) for i in aai_output[1][i:i + 531]]
            aai_feature.append(temp)
        for i in range(0, len(blo_output[1]), 20):
            temp = blo_output[1][i:i + 20]
            blo_feature.append(temp)
        for i in range(0, len(zsl_output[1]), 5):
            temp = zsl_output[1][i:i + 5]
            zsl_feature.append(temp)
        aa_fea_matrx = np.hstack([np.array(bin_feature), np.array(blo_feature), np.array(zsl_feature), np.array(aai_feature)])
        aa_feature_list.append([feature_id, aa_fea_matrx])
    aa_feature_list = np.array(aa_feature_list, dtype=object)
    print(aa_feature_list)
    np.save(temp_file_path, aa_feature_list)
    return temp_file_path + '.npy'

if __name__ == '__main__':
    feature_generator('/Users/ggcl7/Desktop/AIP_data/data/train.txt', '/Users/ggcl7/Desktop/AIP_data/data/train-all')
