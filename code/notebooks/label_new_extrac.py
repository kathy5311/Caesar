#!/usr/bin/python 3
import numpy as np
import os
import sys
def make_one(log, outf):
    label ={}
    for l in open(log):
        if l.strip()[0].islower() or l.strip()[0].isupper(): continue
        words =l[:-1].split()
        #if len(words) != 6: continue
        #print(words)
        reschain = words[1]+" "+ words[2]+words[3]
        P = float(words[5])
        label[reschain]=P
        #print(reschain)
        #print(P)

    np.savez(outf, label=label)

log_path = "/home/kathy531/Caesar/data/log/"
log = os.path.join(log_path,sys.argv[1])
outf_path = '/home/kathy531/Caesar/data/npz0729/'
outf = os.path.join(outf_path, os.path.splitext(os.path.basename(log))[0]+'.label.npz')

make_one(log, outf)

