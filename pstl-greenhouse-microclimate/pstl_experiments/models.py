from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    """Portable reference selective SSM. Δ, B and C depend on each input token."""
    def __init__(self,d_model,d_state,dropout=0.1):
        super().__init__(); self.d_model=d_model; self.d_state=d_state
        self.in_proj=nn.Linear(d_model,2*d_model); self.conv=nn.Conv1d(d_model,d_model,3,padding=2,groups=d_model)
        self.dt_proj=nn.Linear(d_model,d_model); self.B_proj=nn.Linear(d_model,d_state); self.C_proj=nn.Linear(d_model,d_state)
        # Stable negative diagonal continuous-time dynamics.
        self.log_A=nn.Parameter(torch.log(torch.arange(1,d_state+1,dtype=torch.float32)).repeat(d_model,1))
        self.D=nn.Parameter(torch.ones(d_model)); self.out_proj=nn.Linear(d_model,d_model); self.drop=nn.Dropout(dropout)
        self.last_update=None
    def forward(self,x,capture=False):
        # x: [batch, time, channel]
        residual=x; u,z=self.in_proj(x).chunk(2,dim=-1)
        u=self.conv(u.transpose(1,2))[...,:x.shape[1]].transpose(1,2); u=F.silu(u)
        delta=F.softplus(self.dt_proj(u)).clamp(max=10.0)
        Bv=self.B_proj(u); Cv=self.C_proj(u); A=-torch.exp(self.log_A.float())
        h=x.new_zeros((x.shape[0],self.d_model,self.d_state)); ys=[]; changes=[]
        for t in range(x.shape[1]):
            dt=delta[:,t].unsqueeze(-1)
            h_new=torch.exp(dt*A.unsqueeze(0))*h + dt*Bv[:,t].unsqueeze(1)*u[:,t].unsqueeze(-1)
            if capture: changes.append(torch.linalg.vector_norm(h_new-h,ord=2,dim=(-2,-1))/(self.d_model*self.d_state)**0.5)
            yt=(h_new*Cv[:,t].unsqueeze(1)).sum(-1)+self.D*u[:,t]
            ys.append(yt); h=h_new
        y=torch.stack(ys,dim=1)*F.silu(z)
        if capture: self.last_update=torch.stack(changes,dim=1).detach()
        return residual+self.drop(self.out_proj(y))

class SpatialAttention(nn.Module):
    def __init__(self,d_model,n_heads,dropout):
        super().__init__(); self.qkv=nn.Linear(d_model,3*d_model); self.proj=nn.Linear(d_model,d_model)
        self.n_heads=n_heads; self.dh=d_model//n_heads; self.coord=nn.Sequential(nn.Linear(3,d_model),nn.Tanh())
        self.bias_scale=nn.Parameter(torch.tensor(1.0)); self.norm=nn.LayerNorm(d_model); self.drop=nn.Dropout(dropout)
    def forward(self,x,coords):
        # Independent spatial attention at each history time. coords are [node, x,y,height].
        b,t,n,d=x.shape; z=self.norm(x)+self.coord(coords)[None,None,:,:]
        q,k,v=self.qkv(z).chunk(3,-1)
        def split(a): return a.reshape(b,t,n,self.n_heads,self.dh).permute(0,1,3,2,4)
        q,k,v=map(split,(q,k,v)); score=(q@k.transpose(-1,-2))/math.sqrt(self.dh)
        xy=coords[:,None,:]-coords[None,:,:]
        dist=torch.linalg.vector_norm(xy,dim=-1)
        bias=-F.softplus(self.bias_scale)*dist
        score=score+bias[None,None,None,:,:]
        att=score.softmax(-1); out=(att@v).permute(0,1,3,2,4).reshape(b,t,n,d)
        return x+self.drop(self.proj(out))

class PSTL(nn.Module):
    def __init__(self,input_dim,n_nodes,n_targets,horizon,coords,d_model=96,d_state=16,n_layers=2,n_heads=4,dropout=.1):
        super().__init__(); self.n_nodes=n_nodes; self.n_targets=n_targets; self.horizon=horizon
        self.coords=torch.as_tensor(coords,dtype=torch.float32)
        self.input=nn.Linear(input_dim,d_model); self.blocks=nn.ModuleList([SelectiveSSM(d_model,d_state,dropout) for _ in range(n_layers)])
        self.spatial=SpatialAttention(d_model,n_heads,dropout); self.norm=nn.LayerNorm(d_model)
        self.head=nn.Sequential(nn.Linear(d_model,d_model),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_model,horizon*n_targets))
    def forward(self,x,capture=False):
        # [batch,time,node,feature]
        b,t,n,f=x.shape
        z=self.input(x).permute(0,2,1,3).reshape(b*n,t,-1)
        for layer in self.blocks: z=layer(z,capture=capture)
        z=z.reshape(b,n,t,-1).permute(0,2,1,3)
        z=self.spatial(z,self.coords.to(x.device)); last=self.norm(z[:,-1])
        return self.head(last).reshape(b,n,self.horizon,self.n_targets).permute(0,2,1,3)
    def state_updates(self):
        vals=[m.last_update for m in self.blocks if m.last_update is not None]
        return torch.stack(vals,dim=0) if vals else None

class Baseline(nn.Module):
    """Small, shape-matched reference families; configs must identify these as implemented variants."""
    def __init__(self,name,input_dim,n_nodes,n_targets,horizon,coords,d_model=96,n_heads=4,dropout=.1):
        super().__init__(); self.name=name; self.n_nodes=n_nodes; self.n_targets=n_targets; self.horizon=horizon
        self.n_heads=n_heads
        self.input=nn.Linear(input_dim,d_model); self.coords=torch.as_tensor(coords,dtype=torch.float32)
        self.coord_proj=nn.Linear(3,d_model)
        if name=='cnn_lstm': self.encoder=nn.LSTM(d_model,d_model,batch_first=True,num_layers=2,dropout=dropout)
        elif name=='transformer':
            self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model,n_heads,4*d_model,dropout,batch_first=True,norm_first=True),num_layers=3)
        elif name in ('gcn','gconv','gat'):
            self.encoder=nn.ModuleList([nn.Linear(d_model,d_model) for _ in range(2)])
            if name=='gconv': self.edge_score=nn.Sequential(nn.Linear(3,16),nn.Tanh(),nn.Linear(16,1))
            if name=='gat': self.q=nn.Linear(d_model,d_model); self.k=nn.Linear(d_model,d_model); self.v=nn.Linear(d_model,d_model)
        else: raise ValueError(f'Unknown baseline {name}')
        self.head=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,d_model),nn.GELU(),nn.Linear(d_model,horizon*n_targets))
    def forward(self,x):
        b,t,n,f=x.shape; z=self.input(x).permute(0,2,1,3).reshape(b*n,t,-1)
        if self.name=='cnn_lstm': z,_=self.encoder(z); z=z[:,-1]
        elif self.name=='transformer': z=self.encoder(z); z=z[:,-1]
        else:
            z=z.mean(1).reshape(b,n,-1)+self.coord_proj(self.coords.to(x.device))[None,:,:]
            # Relative coordinates are explicit; no regular-grid reshape/convolution is used.
            co=self.coords.to(x.device); rel=co[None,:,:]-co[:,None,:]
            d=torch.linalg.vector_norm(rel,dim=-1)
            if self.name=='gcn':
                sigma=d[d>0].median().clamp_min(1e-4); adj=torch.exp(-(d/sigma)**2); adj=adj/(adj.sum(-1,keepdim=True)+1e-8)
                for layer in self.encoder: z=F.gelu(layer(adj@z))+z
            elif self.name=='gconv':
                adj=self.edge_score(rel).squeeze(-1).softmax(-1)
                for layer in self.encoder: z=F.gelu(layer(adj@z))+z
            else:
                for layer in self.encoder:
                    q=self.q(z).reshape(b,n,self.n_heads,-1).transpose(1,2)
                    k=self.k(z).reshape(b,n,self.n_heads,-1).transpose(1,2)
                    v=self.v(z).reshape(b,n,self.n_heads,-1).transpose(1,2)
                    att=(q@k.transpose(-1,-2))/math.sqrt(q.shape[-1]); att=att.softmax(-1)
                    msg=(att@v).transpose(1,2).reshape(b,n,-1); z=F.gelu(layer(msg))+z
            z=z.reshape(b*n,-1)
        return self.head(z).reshape(b,n,self.horizon,self.n_targets).permute(0,2,1,3)

def build_model(name,input_dim,n_nodes,n_targets,horizon,coords,cfg):
    kw=dict(input_dim=input_dim,n_nodes=n_nodes,n_targets=n_targets,horizon=horizon,coords=coords,
            d_model=int(cfg.get('d_model',96)),dropout=float(cfg.get('dropout',.1)))
    if name=='pstl': return PSTL(**kw,d_state=int(cfg.get('d_state',16)),n_layers=int(cfg.get('n_layers',2)),n_heads=int(cfg.get('n_heads',4)))
    return Baseline(name,**kw,n_heads=int(cfg.get('n_heads',4)))
