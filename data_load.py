from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
import sys


import torch
from torch.autograd import Variable
from torch import optim
from sklearn.model_selection import train_test_split
import itertools



def convert_seq_to_bicoding(seq,wsize=101):
    #return bicoding for a sequence
    seq=seq.replace('U','T') #turn rna seq to dna seq if have
    feat_bicoding=[]
    bicoding_dict={'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],'N':[0,0,0,0]}
    if len(seq)<101:
        seq=seq+'N'*(101-len(seq))
    for each_nt in seq:
        feat_bicoding+=bicoding_dict[each_nt]
    return feat_bicoding



def load_data_bicoding(in_fa):
    data=[]
    for record in SeqIO.parse(in_fa, "fasta"):
        seq=str(record.seq)
        bicoding=convert_seq_to_bicoding(seq)
        data.append(bicoding)
    #print(len(data))

    return data



def load_data_bicoding_with_header(in_fa):
    data=[]
    fa_header=[]
    for record in SeqIO.parse(in_fa, "fasta"):
        seq=str(record.seq)
        bicoding=convert_seq_to_bicoding(seq)
        data.append(bicoding)
        fa_header.append(str(record.description))
    #print(len(data))

    return data, fa_header






def load_train_test_bicoding(pos_train_fa,neg_train_fa,pos_test_fa,neg_test_fa):
    data_pos_train = []
    data_neg_train = []
    data_pos_test = []
    data_neg_test = []

    data_pos_train = load_data_bicoding(pos_train_fa)
    data_neg_train = load_data_bicoding(neg_train_fa)
    data_pos_test = load_data_bicoding(pos_test_fa)
    data_neg_test = load_data_bicoding(neg_test_fa)

    data_train = np.array([_ + [1] for _ in data_pos_train] + [_ + [0] for _ in data_neg_train])
    np.random.seed(42)
    np.random.shuffle(data_train)

    X_train = np.array([_[:-1] for _ in data_train])
    y_train = np.array([_[-1] for _ in data_train])

    X_test = np.array([_ for _ in data_pos_test] + [_ for _ in data_neg_test])
    y_test = np.array([1 for _ in data_pos_test] + [0 for _ in data_neg_test])
    #print(X_train.shape[1])


    return X_train,y_train,X_test,y_test



def load_train_bicoding(pos_train_fa,neg_train_fa):
    data_pos_train = []
    data_neg_train = []

    data_pos_train = load_data_bicoding(pos_train_fa)
    data_neg_train = load_data_bicoding(neg_train_fa)


    data_train = np.array([_ + [1] for _ in data_pos_train] + [_ + [0] for _ in data_neg_train])
    np.random.seed(42)
    np.random.shuffle(data_train)

    X_train = np.array([_[:-1] for _ in data_train])
    y_train = np.array([_[-1] for _ in data_train])


    return X_train,y_train



def load_train_val_bicoding(pos_train_fa,neg_train_fa):
    data_pos_train = []
    data_neg_train = []

    data_pos_train = load_data_bicoding(pos_train_fa)
    data_neg_train = load_data_bicoding(neg_train_fa)


    data_train = np.array([_ + [1] for _ in data_pos_train] + [_ + [0] for _ in data_neg_train])
    np.random.seed(42)
    np.random.shuffle(data_train)

    X = np.array([_[:-1] for _ in data_train])
    y = np.array([_[-1] for _ in data_train])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0/8, random_state=42)

    return X_train,y_train,X_test,y_test




def load_test_bicoding(pos_test_fa,neg_test_fa):
    data_pos_test = []
    data_neg_test = []

    data_pos_test = load_data_bicoding(pos_test_fa)
    data_neg_test = load_data_bicoding(neg_test_fa)

    X_test = np.array([_ for _ in data_pos_test] + [_ for _ in data_neg_test])
    y_test = np.array([1 for _ in data_pos_test] + [0 for _ in data_neg_test])
    #print(X_train.shape[1])


    return X_test,y_test



def load_in_torch_fmt(X_train, y_train, X_test, y_test,vec_len):
    X_train = X_train.reshape(X_train.shape[0], int(X_train.shape[1]/vec_len), vec_len)
    X_test = X_test.reshape(X_test.shape[0], int(X_test.shape[1]/vec_len), vec_len)
    #print(X_train.shape)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    #y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    print('test')