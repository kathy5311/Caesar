import os
import numpy as np
import sys
from multiprocessing import Pool
"""
def process_file(file):
    if len(file)> 8:
        return
    else:
        file_pdb=[i.lstrip("\n") for i in open('/ml/PDBbind/refined-set/'+file[:4]+'/'+file[:4]+'_protein.pdb')]
        check_list=[]
        with open("train_list_amino.txt","a") as f:
            for line in file_pdb:
                if line[:4]=="ATOM":
                    if line[22:26] not in check_list:
                        check_list.append(line[22:26])
                        data=(file[:4]+'.'+line[17:20].strip()+" "+line[21].strip()+line[22:26].strip())
                        print(data)
                        f.write(data+("\n"))

if __name__ == "__main__":
    file_li = [l.strip("\n") for l in open('label_log_file.txt')]

    with Pool() as pool:
        pool.map(process_file, file_li)
"""
file_li=[l.strip("\n") for l in open('prop_label.txt')]
for file in file_li:

    if len(file)>8:
        break
    else:
    

        file_pdb=[i.lstrip("\n") for i in open('/home/kathy531/Caesar/data/PDB/'+file+'/_0001.pdb')]
       # print(file_pdb)
        #print(i.strip("\n"))
        check_list=[]
        with open("train_list.txt","a") as f:
            for line in file_pdb:
                if line[:4]=="ATOM":
                    if line[22:26] not in check_list:
                        check_list.append(line[22:26])
                        data=(file+'.'+line[17:21].strip()+" "+line[21].strip()+line[22:26].strip())
                        print(data)
                        f.write(data+("\n"))
                   # print(insert)
f.close()

 # with open("train_list.txt", "a") as f:
