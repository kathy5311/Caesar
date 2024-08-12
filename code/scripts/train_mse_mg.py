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

    ## loss
    train_loss_empty={"total":[],"loss1":[]}
    valid_loss_empty={"total":[],"loss1":[]}

    epoch=0
    optimizer=torch.optim.Adam(model.parameters(),lr=args_in.LR,weight_decay=1e-5)

    if os.path.exists("/home/kathy531/Caesar/code/scripts/models/%s/model.pkl"%args_in.modelname):
        if not silent: print("Loading a checkpoint")
        checkpoint = torch.load(os.path.join("/home/kathy531/Caesar/code/scripts/models", args_in.modelname, "model.pkl"),map_location=device)

        trained_dict = {}
        model_dict = model.state_dict()
        model_keys = list(model_dict.keys())

        for key in checkpoint["model_state_dict"]:
            if key in model_keys:
                wts = checkpoint["model_state_dict"][key]
                trained_dict[key] = wts
            else:
                print("skip", key)

        model.load_state_dict(trained_dict, strict=False)

        epoch = checkpoint["epoch"]+1
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        if not silent: print("Restarting at epoch", epoch)

    else:
        if not silent and rank == 0: print("Training a new model")
        train_loss = train_loss_empty
        valid_loss = valid_loss_empty

        modelpath = os.path.join("/home/kathy531/Caesar/code/scripts/models", args_in.modelname)
        if not os.path.isdir(modelpath):
            if not silent: print("Creating a new dir at", modelpath)
            os.mkdir(modelpath)

    if not silent and rank == 0:
        nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Nparams:", nparams)

    return model,optimizer,epoch,train_loss,valid_loss


def load_data(dataf, world_size, rank):
    loader_params = {
        'shuffle': False, #should be False with ddp
        'num_workers': 10,  # num_workers 조정-> 여러 실험 방법 존재. 노션 참조
        'pin_memory': True,
        'collate_fn': collate,
        'batch_size': args.nbatch,
        'worker_init_fn': np.random.seed()
    }

    trainset = DataSet(args.dataf_train, args)
    validset = DataSet(args.dataf_valid, args)

    ### added
    sampler_t = torch.utils.data.distributed.DistributedSampler(trainset,num_replicas=world_size,rank=rank)
    sampler_v = torch.utils.data.distributed.DistributedSampler(validset,num_replicas=world_size,rank=rank)

    train_loader =torch.utils.data.DataLoader(trainset, sampler=sampler_t, **loader_params)
    valid_loader =torch.utils.data.DataLoader(validset, sampler=sampler_v, **loader_params)
    ###

    return train_loader, valid_loader

def run_an_epoch(model, optimizer, data_loader, train, rank=0, verbose=False):
    import sys
    lossfunc = torch.nn.CrossEntropyLoss()
    lossfunc_mse = torch.nn.MSELoss()

    loss_tmp = {'total':[],'loss1':[]}
    device=torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")
    nerr = 0

    '''for i in data_loader:
        print(i)
        #print(data_loader[i])
        print()'''

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
                    do_dropout=train #dropout꺼주기
                    pred_test=model(G.to(device), do_dropout)
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

                    loss1 = lossfunc(Ssum,label)
                    Ssum_soft=F.softmax(Ssum,dim=1)

                    range_tensor=torch.tensor([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65,1.75,1.85,1.95,2.05,2.15,2.25,2.35,2.45,2.55])
                    range_tensor=range_tensor.to(device)
                    expected=torch.sum(Ssum_soft * range_tensor, dim=1)
                    expected=expected.to(device)

                    loss2 = lossfunc_mse(expected, label_int.float())
                    loss = loss1 + loss2


                    if train:
                        if (torch.isnan(loss).sum() == 0): #adidng no nan values
                            optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                            optimizer.step()
                    else:
                        pass

                    if verbose:
                        verbl = f'Rank {rank}:'
                        for tag,l,ex,S in zip(info['target'], label_int, expected, Ssum_soft):
                            MSE=abs(ex.item()- l.item())**2
                            verbl += f"{tag:9s} {l.item():.4f} {ex.item():.4f} {torch.max(S):.4f} {MSE:.4f} |"
                        print(verbl)

                        '''
                        for tag,S,l in zip(info['target'], label_int, expected):

                        verbl += f"{tag:9s} {S:.4f} {l:.4f} | "
                        if torch.isnan(S).any() or torch.isnan(label).any():
                        print("S: ",torch.isnan(S).any())
                        print("label: ",torch.isnan(label).any())
                        print(tag)
                        sys.exit()
                        '''

                    loss_tmp['loss1'].append(loss1.cpu().detach().numpy())
                    loss_tmp['total'].append(loss.cpu().detach().numpy())
                else:
                    continue
    return loss_tmp

def main( rank, world_size, dumm ):
    #nan detect function
    #torch.autograd.set_detect_anomaly(True)

    ### add or modified
    gpu = rank%world_size
    dist.init_process_group(backend='gloo',world_size=world_size,rank=rank) #gloo, NCCL -> backend type (NCCL only when building with CUDA)

    model,optimizer,init_epoch,train_loss,valid_loss = load_model(args, rank)
    train_loader, valid_loader = load_data(args, world_size, rank)

    model = DDP(model,device_ids=[gpu],find_unused_parameters=False)
    ###

    for epoch in range(init_epoch,args.maxepoch):
        if rank == 0: print("epoch:", epoch)
        # if epoch==init_epoch+1:
           # break
        model.train()
        loss_t =run_an_epoch(model, optimizer, train_loader, True, rank, verbose=True)

        for key in train_loss:
            train_loss[key].append(np.array(loss_t[key]))

        #model.eval()
        with torch.no_grad():
            model.eval()
            loss_v = run_an_epoch(model, optimizer, valid_loader, False, rank,
                                  (args.verbose and epoch == args.maxepoch-1) )

        for key in valid_loss:
            valid_loss[key].append(np.array(loss_v[key]))




        print("Train/Valid: %3d %8.4f %8.4f"%(epoch, float(np.mean(loss_t['total'])),
                                              float(np.mean(loss_v['total']))))


        if np.min([np.mean(vl) for vl in valid_loss["total"]]) == np.mean(valid_loss["total"][-1]):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
            }, os.path.join("/home/kathy531/Caesar/code/scripts/models", args.modelname, "best.pkl"))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
        }, os.path.join("/home/kathy531/Caesar/code/scripts/models", args.modelname, "model.pkl"))


if __name__ == "__main__":
    mp.freeze_support() #multi process에서는 이 메인 구문을 실행하지 않겠다.
    torch.set_num_threads( int( os.getenv('SLURM_CPUS_PER_TASK', 4) ) )
    world_size=torch.cuda.device_count()
    print("Using %d GPUs.."%world_size)

    if ('MASTER_ADDR' not in os.environ):
        os.environ['MASTER_ADDR'] = 'localhost' # multinode requires this set in submit script
    if ('MASTER_PORT' not in os.environ):
        os.environ['MASTER_PORT'] = '12346'

    mp.spawn(main,args=(world_size,0),nprocs=world_size,join=True)
    
    # validation 할 때 주의 해야함
    
