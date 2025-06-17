import numpy as np

nrandn=np.random.randn
nrand=np.random.rand
nrandint=np.random.randint
nzeros=np.zeros
nones=np.ones
neye=np.eye
neinsum=np.einsum
ncat=np.concatenate

nuns=np.expand_dims
nsqu=np.squeeze
nlinspace=np.linspace
narange=np.arange

nintersect1d=np.intersect1d
def nintersect1dmore(arrays):
    from functools import reduce
    return reduce(nintersect1d,arrays)

nunion1d=np.union1d
def nunion1dmore(arrays):
    from functools import reduce
    return reduce(nunion1d,arrays)