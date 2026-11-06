import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

HERE=Path(__file__).resolve().parent


def encoder(pretrained=True):
    model=resnet18(weights=None)
    if pretrained:
        model.load_state_dict(torch.load(HERE/'resnet18-f37072fd.pth',map_location='cpu',weights_only=True))
    return model


def stages(model,x):
    x=model.maxpool(model.relu(model.bn1(model.conv1(x))))
    a=model.layer1(x)
    b=model.layer2(a)
    c=model.layer3(b)
    d=model.layer4(c)
    return a,b,c,d


class ConvBlock(nn.Sequential):
    def __init__(self,cin,cout):
        super().__init__(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.SiLU(),
                         nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.SiLU())


class LandmarkNet(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        self.enc=encoder(pretrained)
        self.enc.fc=nn.Identity()
        self.dec3=ConvBlock(512+256,128)
        self.dec2=ConvBlock(128+128,96)
        self.dec1=ConvBlock(96+64,64)
        self.heatmap=nn.Conv2d(64,1,1)

    def forward(self,x):
        a,b,c,d=stages(self.enc,x)
        x=self.dec3(torch.cat([F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
        x=self.dec2(torch.cat([F.interpolate(x,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        x=self.dec1(torch.cat([F.interpolate(x,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return self.heatmap(x)[:,0]

    def train(self,mode=True):
        super().train(mode)
        for layer in self.enc.modules():
            if isinstance(layer,nn.BatchNorm2d):
                layer.eval()
        return self


def normalize(x):
    mean=x.new_tensor([.485,.456,.406]).view(1,3,1,1)
    std=x.new_tensor([.229,.224,.225]).view(1,3,1,1)
    return ((x-mean)/std).contiguous(memory_format=torch.channels_last)


def augment(x,points):
    b,_,h,w=x.shape
    angle=(torch.rand(b,device=x.device)-.5)*(24*math.pi/180)
    scale=.9+.2*torch.rand(b,device=x.device)
    flip=torch.where(torch.rand(b,device=x.device)<.5,-1.,1.)
    theta=torch.zeros((b,2,3),device=x.device)
    c,s=angle.cos()/scale,angle.sin()/scale
    theta[:,0,0],theta[:,0,1]=c*flip,-s
    theta[:,1,0],theta[:,1,1]=s*flip,c
    theta[:,:,2]=(torch.rand(b,2,device=x.device)-.5)*.10
    grid=F.affine_grid(theta,x.shape,align_corners=False)
    out=F.grid_sample(x,grid,padding_mode='zeros',align_corners=False)
    p=2*(points+.5)/points.new_tensor([w,h])-1
    transformed=torch.linalg.solve(theta[:,:,:2],(p-theta[:,:,2]).unsqueeze(-1)).squeeze(-1)
    transformed=(transformed+1)*points.new_tensor([w,h])/2-.5
    gamma=.85+.3*torch.rand((b,1,1,1),device=x.device)
    gains=.85+.3*torch.rand((b,1,1,1),device=x.device)
    color=.94+.12*torch.rand((b,3,1,1),device=x.device)
    out=(out.clamp(0,1).pow(gamma)*gains*color).clamp(0,1)
    return out,transformed


def coordinates(logits,input_side=512,local=False):
    b,h,w=logits.shape
    values=logits.float()
    yy,xx=torch.meshgrid(torch.arange(h,device=values.device),torch.arange(w,device=values.device),indexing='ij')
    if local:
        maximum=values.flatten(1).argmax(1)
        cy,cx=maximum//w,maximum%w
        mask=(xx[None]-cx[:,None,None]).abs()<=5
        mask=mask & ((yy[None]-cy[:,None,None]).abs()<=5)
        values=values.masked_fill(~mask,-1e4)
    p=values.flatten(1).softmax(1).view(b,h,w)
    x=(p*xx).sum((1,2))
    y=(p*yy).sum((1,2))
    return (torch.stack([x,y],1)+.5)*input_side/w-.5


def heatmap_loss(logits,points,input_side=512,sigma=2.):
    b,h,w=logits.shape
    yy,xx=torch.meshgrid(torch.arange(h,device=logits.device),torch.arange(w,device=logits.device),indexing='ij')
    center=(points+.5)*(w/input_side)-.5
    target=torch.exp(-((xx[None]-center[:,0,None,None])**2+(yy[None]-center[:,1,None,None])**2)/(2*sigma*sigma))
    target=target/(target.sum((1,2),keepdim=True).clamp_min(1e-8))
    ce=-(target.flatten(1)*logits.float().flatten(1).log_softmax(1)).sum(1).mean()
    coords=coordinates(logits,input_side,False)
    coord_loss=((coords-points)/input_side).square().mean()
    return ce+20*coord_loss


def restore(points,cache,indices):
    indices=np.asarray(indices)
    return (points-cache['pads'][indices]+.5)/cache['scales'][indices]-.5+cache['crops'][indices,:2]


def rmse(pred,target):
    return float(np.sqrt(np.mean(np.square(np.asarray(pred)-np.asarray(target)))))


@torch.no_grad()
def update_ema(ema,model,decay=.75):
    for a,b in zip(ema.parameters(),model.parameters()):
        a.lerp_(b,1-decay)
    for a,b in zip(ema.buffers(),model.buffers()):
        a.copy_(b)


@torch.no_grad()
def predict(model,pixels,indices,batch_size=8,tta=False,amp=True):
    model.eval()
    results={'local':[],'global':[]}
    for start in range(0,len(indices),batch_size):
        index=indices[start:start+batch_size]
        x=pixels[index].float()/255
        with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
            logits=model(normalize(x))
        plain={k:coordinates(logits,x.shape[-1],k=='local') for k in results}
        if tta:
            with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                flipped=model(normalize(x.flip(-1)))
            for k in results:
                point=coordinates(flipped,x.shape[-1],k=='local')
                point[:,0]=x.shape[-1]-1-point[:,0]
                plain[k]=(plain[k]+point)/2
        for k in results:
            results[k].append(plain[k].cpu().numpy())
    return {k:np.concatenate(v) for k,v in results.items()}
