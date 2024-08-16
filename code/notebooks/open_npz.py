import numpy as np
f='/home/kathy531/Caesar/data/npz0729/1a0i_rev.prop.npz'
data=np.load(f, allow_pickle=True)
#print(list(data.keys()))
for i in list(data.keys()):
    print(i)
    print(data[i].shpae)
    print()