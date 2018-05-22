
import sys
import os

from data_load import *
from model_torch import *

import torch
from torch.autograd import Variable
import argparse
import numpy as np
import torch.nn.functional as F

from Bio import SeqIO
from Bio.Seq import Seq


class modelStruc:
    def __init__(self, name, if_rnn):
        paras=name.split('_')
        self.filter_num=paras[0]
        self.filter_size=paras[1]
        self.pool_size=int(paras[2])
        self.cnndrop_out=float(paras[3])
        self.if_bn=paras[4]
        self.if_bce = paras[5]
        self.fc_size=int(paras[6])
        if if_rnn=='Y':
            self.rnn_size=int(paras[-1])



def scores_to_file(score,out_fn):
    with open(out_fn,'w') as f:
        f.write(','.join([ str(_) for _ in score ])+'\n')


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


    args = parser.parse_args()


    in_fa = args.in_fa
    wordvec_len = 4


    model_path = args.model_dir
    checkpoint = torch.load(model_path + '/' + 'checkpoint.pth.tar')
    if 'rnn' in args.mode:
        model_paras = modelStruc(os.path.basename(model_path), 'Y')
    else:
        model_paras = modelStruc(os.path.basename(model_path), 'N')

    n_classes = 2
    loss = torch.nn.CrossEntropyLoss(size_average=False)

    if model_paras.if_bce == 'Y':
        n_classes = n_classes - 1
        loss = torch.nn.BCELoss(size_average=False)

    if args.mode == 'cnn':
        model = ConvNet(output_dim=n_classes, args=model_paras, wordvec_len=wordvec_len)

    if args.mode == 'cnn-rnn':
        model = ConvNet_BiLSTM(output_dim=n_classes, args=model_paras, wordvec_len=wordvec_len)

    print('model loaded')
    model.load_state_dict(checkpoint['state_dict'])


    model.eval()


    with open(args.out_fn, 'w') as f:
        for record in SeqIO.parse(args.in_fa, "fasta"):
            desc = str(record.description)
            seq = str(record.seq)
            X_test = np.array([convert_seq_to_bicoding(seq)])
            X_test = X_test.reshape(X_test.shape[0], int(X_test.shape[1] / wordvec_len), wordvec_len)
            X_test = torch.from_numpy(X_test).float()
            X_test = Variable(X_test, requires_grad=True)


            fx = model.forward(X_test, model_paras)
            if model_paras.if_bce == 'Y':
                pred_prob = F.sigmoid(fx).data.numpy()[0][0]
                y_test = np.array([1])

                y_test = torch.from_numpy(y_test).long()
                y_test = Variable(y_test, requires_grad=False)

                output = loss.forward(F.sigmoid(fx).squeeze(), y_test.type(torch.FloatTensor).squeeze())
            else:
                pred_prob = np.exp(F.log_softmax(fx).data.numpy())[0][1]

                y_test = np.array([1])

                y_test = torch.from_numpy(y_test).long()
                y_test = Variable(y_test, requires_grad=False)

                output = loss.forward(fx, y_test)

            output.backward()

            grads = torch.abs(X_test.grad) * X_test
            grads = grads.data.cpu().numpy()[0]

            scores = []
            for i in range(len(grads)):
                scores.append(max(grads[i]))    #remove zero element
            max_score = max(scores)
            max_score_idx = [i for i, j in enumerate(scores) if j == max_score]


            f.write('#seq_info: '+desc+'\n')
            f.write('#prob: '+str(pred_prob)+'\n')
            f.write('\t'.join(list(seq))+'\n')
            f.write('\t'.join([ str(_) for _ in scores])+'\n')













