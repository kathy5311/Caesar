import sys
import os

# 현재 파일의 디렉토리 경로
current_dir = os.path.dirname(os.path.abspath(__file__))
#print(current_dir)
# src 디렉토리를 경로에 추가
sys.path.append(os.path.join(current_dir, '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.all_atom_model import MyModel
from dataset import DataSet, collate
from args import args_default as args
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from model.all_atom_model import MyModel
from ..src.dataset import DataSet, collate
from ..src.args import args_default as args
'''

## added
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
###

args.modelname='prac_0805_big'
def load_model(args_in,rank=0,silent=False):
    device = torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")
    ## model
    model = MyModel(args_in, device)
    model.to(device)


    if os.path.exists(f"/home/kathy531/Caesar/code/scripts/models/{args_in.modelname}/best.pkl"):
        if not silent: print("Loading the best model checkpoint")
        checkpoint = torch.load(os.path.join("/home/kathy531/Caesar/code/scripts/models", args_in.modelname, "best.pkl"), map_location=device)

        # Remove 'module.' prefix
        state_dict = checkpoint["model_state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' from the key
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        if not silent: print("Model loaded successfully")

    else:
        raise FileNotFoundError(f"No checkpoint found at /home/kathy531/Caesar/code/scripts/models/{args_in.modelname}/best.pkl")

    model.eval()  # Set the model to evaluation mode
    return model


def load_data(dataf, world_size, rank):
    loader_params = {
        'shuffle': False, #should be False with ddp
        'num_workers': 10,  # num_workers 조정-> 여러 실험 방법 존재. 노션 참조
        'pin_memory': True,
        'collate_fn': collate,
        'batch_size': 1,
        'worker_init_fn': np.random.seed()
    }

    validset = DataSet(args.dataf_valid, args)

    sampler_v = torch.utils.data.distributed.DistributedSampler(validset,num_replicas=world_size,rank=rank)

    valid_loader =torch.utils.data.DataLoader(validset, sampler=sampler_v, **loader_params)
    ###

    return valid_loader

def run_an_epoch(model, data_loader, rank=0):

    device=torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")

    for i,(G,label,mask,info,label_int) in enumerate(data_loader):
        if len(label) == 0:
        #    nerr += 1
        #    print('error ', nerr, i)
            continue
        ### added
        with torch.cuda.amp.autocast(enabled=False):
            with model.no_sync(): #should be commented if
        ###
                if type(G) != list:
                    pred_test = model(G.to(device), do_dropout=False)

                    if pred_test==None:
                        continue
                    else:
                        if torch.isnan(pred_test).sum()==0:
                            pred = pred_test.squeeze() #model(G.to(device))
                        elif torch.any(torch.isnan(pred_test)):
                            print("Warning: detected nan, resetting EGNN output to zero")
                            pred=torch.zeros_like(pred_test)
                            pred=pred.squeeze()
                            pred.requires_grad_(True)

                        mask = mask.to(device)

                    #pred와 Mask 속에 nan값 있는지 확인
                    if (torch.isnan(pred).sum()!=0):
                        print("pred nan")

                    if (torch.isnan(mask).sum()!=0):
                        print("mask nan")
                    Ssum = torch.sum(torch.einsum('bij,ik->bjk',mask,pred),dim=1)
                    label = label.to(device)
                    label_int=label_int.to(device)

                    Ssum_soft=F.softmax(Ssum,dim=1)

                    range_tensor=torch.tensor([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65,1.75,1.85,1.95,2.05,2.15,2.25,2.35,2.45,2.55])
                    range_tensor=range_tensor.to(device)
                    ex_val=torch.sum(Ssum_soft * range_tensor, dim=1)
                    ex_val=round(ex_val.item(),4)
                    #ex_val=ex_val.to(device)
                
                    label_int=round(label_int.item(),4)
                    resname,resno=info['target'][0].split(' ')
                    proname,resname=resname.split('.')
                    MSE=round(abs(ex_val-label_int)**2,4)
         

                    print(f"Rank {rank}:", proname, resname, resno, label_int,ex_val,label.item(),torch.argmax(Ssum_soft).item(),MSE)


def main( rank, world_size):
    #nan detect function
    #torch.autograd.set_detect_anomaly(True)

    ### add or modified
    gpu = rank%world_size
    dist.init_process_group(backend='gloo',world_size=world_size,rank=rank) #gloo, NCCL -> backend type (NCCL only when building with CUDA)

    model= load_model(args, rank)
    model = DDP(model,device_ids=[gpu],find_unused_parameters=False)
    valid_loader = load_data(args, world_size, rank)

    ###
    run_an_epoch(model, valid_loader, rank)

if __name__ == "__main__":
    torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK', 4)))
    world_size = torch.cuda.device_count()
    print("Using %d GPUs for inference.." % world_size)

    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12292'

    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)