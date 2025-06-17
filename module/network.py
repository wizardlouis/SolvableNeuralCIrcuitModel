import torch
import torch.nn as nn
from .utils import *
import math

class LeakyRNN(nn.Module):
    g=0.01

    def __init__(self, hyper: Mydict):
        '''
        @param hyper: Mydict,keys in hyper: N,N_I,N_R,N_O,tau,act_func,
        editted 2023.7.13
        '''
        super(LeakyRNN, self).__init__()
        self.hyper = hyper
        self.reinit(hyper)

    def reinit(self, hyper):
        N = hyper.N
        N_I = hyper.N_I
        N_O = hyper.N_O
        N_R = hyper.N_R
        self.tau = hyper.tau
        self.act_func = hyper.act_func
        self.In = nn.Parameter(trandn(N, N_I), requires_grad=True)
        self.Out = nn.Parameter(trandn(N, N_O), requires_grad=True)
        self.h0 = nn.Parameter(torch.zeros(N), requires_grad=False)
        if hyper.N_R == -1:
            self.J = nn.Parameter(LeakyRNN.g*trandn(N, N), requires_grad=True)
            self.param = [self.In, self.Out, self.J, self.h0]
        else:
            self.U = nn.Parameter(LeakyRNN.g*trandn(N, N_R), requires_grad=True)
            self.V = nn.Parameter(LeakyRNN.g*trandn(N, N_R), requires_grad=True)
            self.param = [self.In, self.Out, self.U, self.V, self.h0]

    def trainable(self, I=True, O=True, R=True, h0=False):
        self.In.requires_grad = I
        self.Out.requires_grad = O
        self.h0.requires_grad = h0
        if self.hyper.N_R == -1:
            self.J.requires_grad = R
        else:
            self.U.requires_grad = R
            self.V.requires_grad = R

    def save(self, savepath):
        torch.save(Mydict(hyper=self.hyper, param=self.state_dict()), savepath)

    @staticmethod
    def load(filepath):
        if type(filepath) == str:
            data = torch.load(filepath)
        elif type(filepath) == Mydict:
            data = filepath
        elif type(filepath) == dict:
            data = Mydict(filepath)
        else:
            raise TypeError('Filepath not correct format!!!')
        model = LeakyRNN(data.hyper)
        model.load_state_dict(data.param)
        return model

    def load_vector(self,loadingvector):
        hp=self.hyper
        assert loadingvector.shape==(hp.N,hp.N_I+hp.N_R*2+hp.N_O)
        I_index=list(range(hp.N_I))
        U_index=list(range(hp.N_I,hp.N_I+hp.N_R))
        V_index=list(range(hp.N_I+hp.N_R,hp.N_I+hp.N_R*2))
        O_index=list(range(hp.N_I+hp.N_R*2,hp.N_I+hp.N_R*2+hp.N_O))
        self.In.data=loadingvector[:,I_index]
        self.U.data=loadingvector[:,U_index]
        self.V.data=loadingvector[:,V_index]
        self.Out.data=loadingvector[:,O_index]

    def forward(self, u: torch.Tensor, g_rec=0., dt=20, get_Encoder=False, device='cpu'):
        assert u.ndim == 3
        assert u.shape[2] == self.hyper.N_I
        n_batch, T = u.shape[:2]
        hyper = self.hyper
        alpha = dt / self.tau

        # init hidden state
        h = uns(self.h0, 0).repeat_interleave(n_batch, 0).to(device)
        h_t = uns(h, 1)
        u = u.to(device)

        # recurrent noise
        Noise_rec = g_rec * trandn(n_batch, T, hyper.N, device=device)

        # recurrent dynamic matrix
        if hyper.N_R == -1:
            J = self.J.to(device)
        else:
            J = self.U.to(device) @ self.V.to(device).T

        # update
        for t in range(T):
            h= (1-alpha)*h+alpha*(torch.einsum('uv,bv->bu',J,self.act_func(h))/hyper.N+torch.einsum('ui,bi->bu',self.In.to(device),u[:,t])+Noise_rec[:,t])
            h_t = torch.cat([h_t, uns(h, 1)], 1)

        # readout
        y_t = torch.einsum('no,btn->bto', self.Out.to(device), self.act_func(h_t)) / hyper.N
        if get_Encoder:
            return Mydict(out=y_t, hidden=h_t)
        else:
            return y_t

    def regularization_loss(self, alpha=0.):
        L2 = nn.MSELoss(reduction='mean')
        L1 = nn.L1Loss(reduction='mean')
        if self.hyper.N_R == -1:
            params = [self.In, self.Out, self.J]
        else:
            params = [self.In, self.Out, self.U, self.V]

        def param_reg(L, param):
            return param.requires_grad * L(param, torch.zeros_like(param))

        return sum([(1. - alpha) * param_reg(L2, param) + alpha * param_reg(L1, param) for param in params])


class reparameterlizedRNN(nn.Module):
    # default parameters
    g = .01
    randomsample = 512

    def __init__(self, hyper: Mydict):
        '''
        @param hyper: Mydict,keys in hyper: N_pop,N_I,N_R,N_O,tau,act_func,
        editted 2023.7.13
        '''
        super(reparameterlizedRNN, self).__init__()
        self.device='cpu'
        self.hyper=hyper
        self.act_func = hyper.act_func
        self.tau = hyper.tau
        self.N_pop = hyper.N_pop
        self.N_I = hyper.N_I
        self.N_R = hyper.N_R
        self.N_O = hyper.N_O
        self.N_F = self.N_I + 2 * self.N_R + self.N_O
        # Index of ranks
        self.I_index = list(range(self.N_I))
        self.U_index = list(range(self.N_I, self.N_I + self.N_R))
        self.V_index = list(range(self.N_I + self.N_R, self.N_I + 2 * self.N_R))
        self.O_index = list(range(self.N_I + 2 * self.N_R, self.N_F))
        # Initialization of Statistics hyperparameters:
        self.h0 = nn.Parameter(torch.zeros(self.N_pop, reparameterlizedRNN.randomsample), requires_grad=False)
        self.G = nn.Parameter(torch.zeros(self.N_pop))
        mu = torch.zeros(self.N_pop, self.N_F)
        self.mu_I = nn.Parameter(mu[:, self.I_index])
        self.mu_U = nn.Parameter(mu[:, self.U_index])
        self.mu_V = nn.Parameter(mu[:, self.V_index])
        self.mu_O = nn.Parameter(mu[:, self.O_index])
        if 'N_sample' in hyper.keys():
            self.N_sample=hyper.N_sample
        else:
            self.N_sample=self.N_F
        C = uns(reparameterlizedRNN.g*torch.eye(self.N_F,self.N_sample),0).repeat_interleave(self.N_pop,0)
        self.C_I = nn.Parameter(C[:, self.I_index])
        self.C_U = nn.Parameter(C[:, self.U_index])
        self.C_V = nn.Parameter(C[:, self.V_index])
        self.C_O = nn.Parameter(C[:, self.O_index])
        # Initialization of Masks
        self.Mask_G = torch.ones_like(self.G)
        self.Mask_mu = torch.ones(self.N_pop, self.N_F)
        self.Mask_C = torch.ones(self.N_pop, self.N_F, self.N_sample)

    def to(self,device):
        super(reparameterlizedRNN,self).to(device)
        self.Mask_G=self.Mask_G.to(device)
        self.Mask_mu=self.Mask_mu.to(device)
        self.Mask_C=self.Mask_C.to(device)
        self.device=device
        return self

    def get_state_dict(self,loading=False):
        state_dict=Mydict(
            hyper=self.hyper,
            param=self.state_dict(),
            Mask=[self.Mask_G, self.Mask_mu, self.Mask_C]
        )
        if loading:
            state_dict.loading=self.noise_loading
        return state_dict
    def save(self, savepath):
        torch.save(self.get_state_dict(),savepath)

    @staticmethod
    def load(filepath):
        if type(filepath) == str:
            data = torch.load(filepath)
        elif type(filepath) == Mydict:
            data = filepath
        elif type(filepath) == dict:
            data = Mydict(filepath)
        else:
            raise TypeError('Filepath not correct format!!!')
        model = reparameterlizedRNN(data.hyper)
        model.load_state_dict(data.param)
        # load Mask
        model.Mask_G = data.Mask[0]
        model.Mask_mu = data.Mask[1]
        model.Mask_C = data.Mask[2]
        return model

    def reinit(self, **kwargs):
        '''
        reinitialization of network through additional hyperparameters for specific task requirements
        @param kwargs: keywords
        @return:
        '''
        keys = ['G', 'mu_I', 'mu_U', 'mu_V', 'mu_O', 'C_I', 'C_U', 'C_V', 'C_O']
        n_keys = len(keys)
        layers = [self.G, self.mu_I, self.mu_U, self.mu_V, self.mu_O, self.C_I, self.C_U, self.C_V, self.C_O]
        # reinitialization of trainable layers
        trainable_keys = [key + '_train' for key in keys]
        for idx in range(n_keys):
            if trainable_keys[idx] in kwargs:
                layers[idx].requires_grad = bool(kwargs[trainable_keys[idx]])
        # reinitialization of weights in trainable layers
        sigma_keys = ['g_' + key for key in keys]
        for idx in range(n_keys):
            if sigma_keys[idx] in kwargs:
                layers[idx].data = kwargs[sigma_keys[idx]] * torch.randn(layers[idx].shape)
        # reinitialization of weights through direct given values
        value_keys = ['w_' + key for key in keys]
        for idx in range(n_keys):
            if value_keys[idx] in kwargs:
                layers[idx].data = kwargs[value_keys[idx]]
        # reinitialization of Mask values:
        Mask_keys = ['Mask_' + key for key in ['G', 'mu', 'C']]
        Masks = [self.Mask_G, self.Mask_mu, self.Mask_C]
        for idx in range(3):
            if Mask_keys[idx] in kwargs:
                Masks[idx].data = kwargs[Mask_keys[idx]]

    def trainable(self, train=True):
        for param in [self.G, self.mu_I, self.mu_U, self.mu_V, self.mu_O, self.C_I, self.C_U, self.C_V, self.C_O]:
            param.requires_grad = train

    def set_zero_grad(self,mu=(),C=()):
        '''
        @param mu: set of zero grad indices for mean value
        @param C: set of zero grad indices for cholesky value
        @return:
        '''
        Offset=[
            0,
            self.N_I,
            self.N_I + self.N_R,
            self.N_F - self.N_O,
            self.N_F
        ]

        def layer_sort(index):
            try:
                id_layer = Offset.index(
                    next(num for num in Offset if index < num)
                )-1
                return id_layer
            except:
                raise ValueError('index crossing valid range!!!')

        if len(mu)!=0:
            layers = [self.mu_I, self.mu_U, self.mu_V, self.mu_O]
            for idx in range(len(mu)):
                id_layer = layer_sort(mu[idx][1])
                layers[id_layer].grad[
                    mu[idx][0],
                    mu[idx][1] - Offset[id_layer]
                ]=0.

        if len(C)!=0:
            layers = [self.C_I,self.C_U,self.C_V,self.C_O]
            for idx in range(len(C)):
                id_layer = layer_sort(C[idx][1])
                layers[id_layer].grad[
                    C[idx][0],
                    C[idx][1] - Offset[id_layer],
                    C[idx][2]
                ]=0

    def get_alpha_p(self):
        alpha_p = torch.softmax(self.G, dim=0)
        return alpha_p * self.Mask_G / (sum(alpha_p * self.Mask_G))

    def get_mu(self):
        mu = torch.cat([self.mu_I, self.mu_U, self.mu_V, self.mu_O], 1)
        return mu * self.Mask_mu

    def get_C(self):
        C = torch.cat([self.C_I, self.C_U, self.C_V, self.C_O], 1)
        return C * self.Mask_C

    def get_cov(self):
        cov = torch.einsum('pij,pkj->pik', self.get_C(), self.get_C())
        return cov

    def get_Statistic(self):
        return Mydict(weight=self.get_alpha_p(),
                      mu=self.get_mu(),
                      cov=self.get_cov())

    def get_Overlap(self, reduction_p=True):
        Sta = self.get_Statistic()
        Overlap_p = torch.einsum('pu,pv->puv', Sta.mu, Sta.mu) + Sta.cov
        if reduction_p:
            return torch.einsum('p,puv->uv', Sta.weight, Overlap_p)
        else:
            return Overlap_p

    def reset_noise_loading(self):
        with torch.no_grad():
            noise_loading = torch.randn(self.N_pop, reparameterlizedRNN.randomsample, self.N_sample,
                                             device=self.device)
            # whiten noise
            mean_noise_loading=noise_loading.mean(1,keepdim=True)
            std_noise_loading=noise_loading.std(1,keepdim=True)
            self.noise_loading=(noise_loading-mean_noise_loading)/std_noise_loading

    def get_loading(self):
        '''
        :return: loading vectors of multipopulations with shape (N_pop, n_samples, dynamical dimensions)
        '''
        mu = self.get_mu()
        C = self.get_C()
        loadingvectors = torch.einsum('pnr,pkr->pnk', self.noise_loading, C) + mu.view(self.N_pop, 1,self.N_F)
        return loadingvectors

    def get_axes(self):
        loadingvectors = self.get_loading()
        return Mydict(I=loadingvectors[:, :, self.I_index],
                      U=loadingvectors[:, :, self.U_index],
                      V=loadingvectors[:, :, self.V_index],
                      O=loadingvectors[:, :, self.O_index])

    def sample_LeakyRNN(self,N):
        Statistic = Mydict(weights=tn(self.get_alpha_p()), means=tn(self.get_mu()), C=tn(self.get_C()))
        from .fit_module import GMMresample
        loadingvector=GMMresample(N,Statistic,C=True)
        hyper = Mydict(N=N, N_I=self.hyper.N_I, N_R=self.hyper.N_R, N_O=self.hyper.N_O, tau=self.hyper.tau,
                       act_func=self.hyper.act_func)
        model=LeakyRNN(hyper)
        model.In.data=tt(loadingvector[:,self.I_index])
        model.U.data=tt(loadingvector[:,self.U_index])
        model.V.data=tt(loadingvector[:,self.V_index])
        model.Out.data=tt(loadingvector[:,self.O_index])
        return model

    def forward(self, u: torch.Tensor, h0=None, dt=10, g_rec=0.02, get_Encoder=False,readout_V=False):
        '''
        @param u:
        @param h0:
        @param dt:
        @param g_rec:
        @param get_Encoder:
        @return:
        '''
        assert u.dim() == 3
        assert u.shape[2] == self.N_I
        Batch_Size, T = u.shape[:2]
        if h0 is None:
            hidden0 = self.h0
        else:
            assert h0.shape == (self.N_pop, reparameterlizedRNN.randomsample)
            hidden0 = h0
        hidden = uns(hidden0, 0).repeat_interleave(Batch_Size, 0)
        hidden_t = uns(hidden, 1)
        alpha = dt / self.tau
        alpha_p = self.get_alpha_p()
        Noise_rec = math.sqrt(2 / alpha) * g_rec * torch.randn(Batch_Size,
                                                               T, self.N_pop, reparameterlizedRNN.randomsample,
                                                               device=self.device)
        axes = self.get_axes()
        for t in range(T):
            External = torch.einsum('bi,pni->bpn', u[:, t], axes.I)
            Selection = torch.einsum('p,pnr,bpn->br', alpha_p, axes.V,
                                     self.act_func(hidden)) / reparameterlizedRNN.randomsample
            Recurrence = torch.einsum('br,pnr->bpn', Selection, axes.U)
            hidden = (1 - alpha) * hidden + alpha * (Recurrence + External + Noise_rec[:, t])
            hidden_t = torch.cat([hidden_t, uns(hidden, 1)], 1)
        if not readout_V:
            out_t = torch.einsum('p,pno,btpn->bto', alpha_p, axes.O,
                                 self.act_func(hidden_t)) / reparameterlizedRNN.randomsample
        else:
            out_t = torch.einsum('p,pno,btpn->bto', alpha_p, axes.V,
                                 self.act_func(hidden_t)) / reparameterlizedRNN.randomsample
        if not get_Encoder:
            return out_t
        else:
            return Mydict(out=out_t, hidden=hidden_t)

    def hidden_align_U(self, hidden, with_kappa_I=False, normalize=False):
        '''
        :param normalize: True->change to normalized axes
        '''
        # notice that in this function,hidden must be reshaped as '...pn' to fit _U
        weights = self.get_alpha_p()
        axes = self.get_axes()
        Norm = 1
        if normalize:
            Norm = 1 / 2
        U_pow2_norm_inv = (torch.einsum('p,pnr,pnr->r', weights, axes.U,
                                        axes.U) / reparameterlizedRNN.randomsample) ** (-Norm)
        kappa = torch.einsum(
            'p,pnr,...pn,r->...r',
            weights,
            axes.U,
            hidden,
            U_pow2_norm_inv) / reparameterlizedRNN.randomsample
        if not with_kappa_I:
            return kappa
        else:
            I_pow2_norm_inv = (torch.einsum('p,pni,pni->i', weights, axes.I,
                                            axes.I) / reparameterlizedRNN.randomsample) ** (-Norm)
            kappa_I = torch.einsum(
                'p,pnr,...pn,r->...r',
                weights,
                axes.I,
                hidden,
                I_pow2_norm_inv) / reparameterlizedRNN.randomsample
            return kappa, kappa_I

    def hidden_align_U_non_ortho(self, hidden):
        '''
        A non-orthogonal solution of collective value for (kappa_I,kappa_r) [batch_size,T,dim]
        '''
        axes = self.get_axes()
        _UI = torch.cat((axes.I, axes.U), dim=-1)
        #_UI:[p,n.k]->[pn,k]
        pinv_UI = torch.linalg.pinv(_UI.reshape(-1, _UI.shape[-1]))
        #hidden:[b,t,p,n]->[b,t,pn]
        kappa_f = hidden.reshape(*hidden.shape[:-2], -1) @ pinv_UI.T
        return kappa_f

    def kappa_align_U(self, kappa, kappa_I=None, normalize=False):
        weights = self.get_alpha_p()
        axes = self.get_axes()
        Norm = 0
        if normalize:
            Norm = 1 / 2
        U_pow2_norm_inv = (torch.einsum('p,pnr,pnr->r', weights, axes.U,
                                        axes.U) / reparameterlizedRNN.randomsample) ** (-Norm)
        hidden = torch.einsum('...r,pnr,r->...pn', kappa, axes.U, U_pow2_norm_inv)
        if kappa_I is None:
            return hidden
        else:
            I_pow2_norm_inv = (torch.einsum('p,pni,pni->i', weights, axes.I,
                                            axes.I) / reparameterlizedRNN.randomsample) ** (-Norm)
            hidden += torch.einsum('...i,pni,i->...pn', kappa_I, axes.I, I_pow2_norm_inv)
            return hidden

    def gradient(self, hidden, Input=None):
        axes = self.get_axes()
        # hidden size(Batch_Size,population,N)
        Selection = torch.einsum('p,pnr,...pn->...r', self.get_alpha_p(), axes.V,
                                 self.act_func(hidden)) / reparameterlizedRNN.randomsample
        Recurrence = torch.einsum('...r,pnr->...pn', Selection, axes.U)
        if Input is None:
            gradient = Recurrence
        else:
            External = torch.einsum('...i,pni->...pn', Input, axes.I)
            gradient = Recurrence + External
        return -hidden + gradient

    def regularization_loss(self, alpha=0.):
        params = [param for param in
                  [self.mu_I, self.mu_U, self.mu_V, self.mu_O, self.C_I, self.C_U, self.C_V, self.C_O]]
        L2 = nn.MSELoss(reduction='sum')
        L1 = nn.L1Loss(reduction='sum')
        def param_reg(L, param):
            return param.requires_grad * L(param, torch.zeros_like(param))
        return sum([(1. - alpha) * param_reg(L2, param) + alpha * param_reg(L1, param) for param in params])

    # All I and U should be orthogonal
    def orthogonal_loss(self, I=True, U=True, V=True, O=False, IU=True, IV=False, IO=False):
        Overlap = self.get_Overlap(reduction_p=True)
        def get_offdiag(x):
            return x-torch.diag(x.diag())
        O_I=Overlap[self.I_index][:,self.I_index]
        O_U=Overlap[self.U_index][:,self.U_index]
        O_V=Overlap[self.V_index][:,self.V_index]
        O_O=Overlap[self.O_index][:,self.O_index]
        O_IU = Overlap[self.I_index][:,self.U_index]
        O_IV=Overlap[self.I_index][:,self.V_index]
        O_IO=Overlap[self.I_index][:,self.O_index]
        diagcomp=[(I,O_I),(U,O_U),(V,O_V),(O,O_O)]
        offdiagcomp=[(IU,O_IU),(IV,O_IV),(IO,O_IO)]
        L = nn.L1Loss(reduction='mean')
        return sum([float(label)*L(get_offdiag(comp),torch.zeros_like(comp)) for label,comp in diagcomp])+sum([float(label)*L(comp,torch.zeros_like(comp)) for label,comp in offdiagcomp])

def set_zeromean_model(model: reparameterlizedRNN):
    '''
    @param model: reparameterlizedRNN
    @return: zero mean model
    '''
    for param in [model.mu_I, model.mu_U, model.mu_V, model.mu_O]:
        param.data = torch.zeros_like(param)
        param.requires_grad = False


def set_isotropy_model(model: reparameterlizedRNN, g=1.):
    '''
    @param model: reparameterlizedRNN
    @param g: standard deviation of Statistic of model
    @return: isotrpic model
    '''
    C = uns(g * torch.eye(model.N_F), 0).repeat_interleave(model.N_pop, 0)
    for slides, param in [(list(range(model.N_I)), model.C_I),
                          (list(range(model.N_I, model.N_I + model.N_R)), model.C_U),
                          (list(range(model.N_I + model.N_R, model.N_I+2 * model.N_R)), model.C_V),
                          (list(range(model.N_I + 2 * model.N_R, model.N_F)), model.C_O)]:
        param.data = C[:, slides]
        param.requires_grad = False