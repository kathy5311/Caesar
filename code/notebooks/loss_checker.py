import sys
import torch
import numpy as np

data_path = '/home/kathy531/Caesar/code/scripts/models/'+sys.argv[1]+'/'
f=data_path+'model.pkl'
checkpoint=torch.load(f, map_location=torch.device('cpu'))
a=checkpoint['train_loss']['total']
b=checkpoint['valid_loss']['total']
c=checkpoint['train_loss']['loss1']
d=checkpoint['valid_loss']['loss1']
a_show=[np.mean(col) for col in a]
b_show=[np.mean(col) for col in b]
c_show=[np.mean(col) for col in c]
d_show=[np.mean(col) for col in d]
print(f"train_loss: {a_show}")
print(f"valid_loss: {b_show}")
print(f"train_loss CE: {c_show}")
print(f"valid_loss CE: {d_show}")

