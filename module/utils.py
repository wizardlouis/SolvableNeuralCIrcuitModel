import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import copy

import sys
sys.path.append('../../')
from sxwtoolbox import *

def switch(key, keylist: list, valuelist: list):
    assert key in keylist
    return valuelist[keylist.index(key)]


def create(path):
    if not os.path.exists(path):
        os.makedirs(path)


trandn = torch.randn


def retanh(x): return x.tanh().relu()


def pf(a1, a2): plt.figure(figsize=(a1, a2))


def pfsp(m, n, a1, a2):
    fig, ax = plt.subplots(m, n, figsize=(a1, a2))
    return fig, ax


def uns(x, dim=0)->torch.tensor:
    return x.unsqueeze(dim=dim)


def tn(x): return x.cpu().detach().numpy()


def tt(x, dtype=torch.float, device="cpu"):
    return torch.tensor(x, dtype=dtype, device=device)


#zip Mydict data of conditions and targets into a zipfile
def zip_data(data:Mydict,zip_order=('condition','target'))->list:
    #get param
    if len(zip_order)==0:
        raise KeyError('No keys in zip_order!!!')
    else:
        for key in zip_order:
            assert key in list(data.keys())
        n_trial=len(data[zip_order[0]])
        for key in zip_order:
            assert n_trial==len(data[key])
        data_zipped=[[] for _ in range(n_trial)]
        for key in zip_order:
            tensor_list=data[key]
            for i,tensor in enumerate(tensor_list):
                data_zipped[i].append(tensor)
    return data_zipped

#unzip a zipfile to a Mydict data of conditions and targets
def unzip_data(data:list,unzip_order=('condition','target'))->Mydict:
    if len(unzip_order)==0:
        raise KeyError('No keys in unzip_order!!!')
    else:
        assert len(data[0])==len(unzip_order)
        data_unzipped=Mydict()
        n_trials=len(data)
        for i,key in enumerate(unzip_order):
            if type(data[0][i])==float:
                data_unzipped[key]=[[data[k][i]] for k in range(n_trials)]
            elif type(data[0][i])==np.array:
                data_unzipped[key]=ncat([nuns(data[k][i],0) for k in range(n_trials)],0)
            elif type(data[0][i])==torch.Tensor:
                data_unzipped[key]=tcat([tuns(data[k][i],0) for k in range(n_trials)],0)
            else:
                raise TypeError('data not supported types!!!')
        return data_unzipped

def find(mylist, target):
    indices = [i for i, x in enumerate(mylist) if x == target]
    return indices




def get_steps(timing:Mydict):
    steps=Mydict(dt=timing.dt)
    for key in timing.keys():
        if key != 'dt':
            steps.__setattr__('T'+key[1:],
                              math.floor(timing.__getattribute__(key)/steps.dt))
    return steps

def unitvector(vector,norm='unit',dim=0):
    if norm=='unit':
        return vector/np.expand_dims(np.sqrt((vector**2).sum(dim)),dim)
    elif norm=='element':
        return vector/np.expand_dims(np.sqrt((vector**2).mean(dim)),dim)
    else:
        raise KeyError('norm param not found in existing models!!!')