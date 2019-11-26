
import sys

from data_load import *
from model_torch import *

import torch
from torch.autograd import Variable
import argparse
import numpy as np
import torch.nn.functional as F
import os

from Bio import SeqIO
from Bio.Seq import Seq



class modelStruc:
    def __init__(self, name, if_rnn):
        paras=name.split('_')
        self.filter_num=paras[0]
        self.filter_size = paras[1]
        self.pool_size=int(paras[2])
        self.cnndrop_out=float(paras[3])
        self.if_bn=paras[4]
        self.if_bce = paras[5]
        self.fc_size=int(paras[6])
        if if_rnn=='Y':
            self.rnn_size=int(paras[-1])



def predict(model, x_val,args):
    model.eval() #evaluation mode do not use drop out
    x = Variable(x_val, requires_grad=False)
    output = model.forward(x,args)
    return output


if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    # model option
    parser.add_argument("-m", "--mode", action="store", dest='mode', required=True,choices=['cnn','cnn-rnn'],
                        help="mode")
    parser.add_argument("-infa", "--in_fa", action="store", dest='in_fa', required=True,
                        help="input fa")
    parser.add_argument("-md", "--model_dir", action="store", dest='model_dir', required=True,
                        help="model directory")
    parser.add_argument("-outfn", "--out_fn", action="store", dest='out_fn', required=True,
                        help="output file")
    parser.add_argument("-batch", "--batch_size", action="store", dest='batch_size', default=256, type=int,
                        help="batch size")

    args = parser.parse_args()



    in_fa = args.in_fa
    wordvec_len = 4

    model_path = args.model_dir
    if model_path[-1] == '/':
        model_path = model_path[:-1]
    checkpoint = torch.load(model_path + '/' + 'checkpoint.pth.tar')
    if 'rnn' in args.mode:
        model_paras = modelStruc(os.path.basename(model_path),'Y')
    else:
        model_paras = modelStruc(os.path.basename(model_path), 'N')


    n_classes = 2

    if model_paras.if_bce == 'Y':
        n_classes = n_classes - 1

    if args.mode == 'cnn':
        model = ConvNet(output_dim=n_classes, args=model_paras,wordvec_len=wordvec_len)

    if args.mode == 'cnn-rnn':
        model = ConvNet_BiLSTM(output_dim=n_classes, args=model_paras,wordvec_len=wordvec_len)

    print('model loaded')
    model.load_state_dict(checkpoint['state_dict'])



    X_test, fa_header=load_data_bicoding_with_header(args.in_fa)
    X_test=np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], int(X_test.shape[1] / wordvec_len), wordvec_len)
    X_test = torch.from_numpy(X_test).float()


    i=0
    N=X_test.shape[0]

    with open(args.out_fn, 'w') as fw:

        while i + args.batch_size < N:
            x_batch = X_test[i:i + args.batch_size]
            header_batch=fa_header[i:i + args.batch_size]

            output_test = predict(model, x_batch, model_paras)

            if model_paras.if_bce == 'Y':
                prob_data = F.sigmoid(output_test).data.numpy()
                for m in range(len(prob_data)):
                    fw.write(header_batch[m]+'\t'+str(prob_data[m][0])+'\n')

            else:

                prob_data = F.log_softmax(output_test).data.numpy()
                for m in range(len(prob_data)):
                    fw.write(header_batch[m]+'\t'+str(np.exp(prob_data)[m][1]) + '\n')

            i += args.batch_size


        x_batch = X_test[i:N]
        header_batch = fa_header[i:N]

        output_test = predict(model, x_batch, model_paras)

        if model_paras.if_bce == 'Y':
            prob_data = F.sigmoid(output_test).data.numpy()
            for m in range(len(prob_data)):
                fw.write(header_batch[m]+'\t'+str(prob_data[m][0]) + '\n')

        else:
            prob_data = F.log_softmax(output_test).data.numpy()
            for m in range(len(prob_data)):
                fw.write(header_batch[m]+'\t'+str(np.exp(prob_data)[m][1]) + '\n')










