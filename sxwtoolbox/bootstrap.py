# -*- codeing = utf-8 -*-
# @time:4/15/24 7:12 PM
# Author:Xuewen Shen
# @File:bootstrap.py
# @Software:PyCharm
import tqdm
import random

def bootstrap(func,data,*args,n_bootstrap=64,disable=True,**kwargs):
    results=[]
    n_samples=len(data)
    for _ in tqdm.tqdm(range(n_bootstrap),disable=disable):
        Index=random.choices(list(range(n_samples)),k=n_samples)
        data_bootstrap=[data[index] for index in Index]
        results.append(func(data_bootstrap,*args,**kwargs))
    return results