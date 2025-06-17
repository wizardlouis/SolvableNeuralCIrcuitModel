# easy tool for torch and alias
import torch
import torch.nn as nn

### alias

trandn=torch.randn
trand=torch.rand
trandint=torch.randint
tzeros=torch.zeros
tones=torch.ones
teye=torch.eye
teinsum=torch.einsum
tcat=torch.cat
tlinspace=torch.linspace
tarange=torch.arange


### easy tools
def tt(x, dtype=torch.float, device="cpu"):
    return torch.tensor(x, dtype=dtype, device=device)

def tn(x): return x.cpu().detach().numpy()

def tuns(x, dim=0)->torch.tensor:
    return x.unsqueeze(dim=dim)

def tsqu(x,dim=0)->torch.tensor:
    return x.unsqueeze(dim=dim)


class BaseModel(nn.Module):
    def __init__(self,hyper,device='cpu'):
        super(BaseModel,self).__init__()
        self.hyper=hyper
        self.device=device

    def to(self,device):
        self.device=device
        super().to(device)