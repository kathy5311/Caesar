import os
import numpy as np
import sys
from multiprocessing import Pool

def process_file(file):
    if len(file) > 8:
        return
    else:
        file_path = f'/home/kathy531/Caesar/data/PDB/{file}_0001.pdb'
        if os.path.exists(file_path):
            file_pdb = [i.lstrip("\n") for i in open(file_path)]
            check_list = []
            data_list = []
            for line in file_pdb:
                if line[:4] == "ATOM":
                    if line[22:26] not in check_list:
                        check_list.append(line[22:26])
                        data = (file + '.' + line[17:21].strip() + " " + line[21].strip() + line[22:26].strip())
                        data_list.append(data)
            return data_list
        else:
            print(f"File not found: {file_path}")
            return []

def save_to_file(data_list):
    with open("train_list.txt", "a") as f:
        for data in data_list:
            f.write(data + "\n")

if __name__ == "__main__":
    file_li = [l.strip("\n") for l in open('prop_label.txt')]
    
    with Pool(40) as pool:
        results = pool.map(process_file, file_li)
    
    # Flatten the list of results and save to file
    flat_list = [item for sublist in results if sublist for item in sublist]
    save_to_file(flat_list)