import argparse
import copy
import csv
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from models import LandmarkNet,encoder,stages,normalize,augment,heatmap_loss,coordinates,predict,restore,rmse,update_ema

HERE=Path(__file__).resolve().parent
OUTPUT=HERE.parent.parent/'outputs'/'task3'


def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    assert torch.cuda.is_available(),'Local CUDA GPU required'


def load():
    d=np.load(HERE/'data.npz')
    pixels=torch.from_numpy(d['images'].copy()).permute(0,3,1,2).contiguous(memory_format=torch.channels_last).cuda()
    return d,pixels


@torch.no_grad()
def extract(model,pixels,flip=False,batch=8):
    model.eval()
    output=[]
    for start in range(0,len(pixels),batch):
        x=pixels[start:start+batch].float()/255
        if flip:
            x=x.flip(-1)
        with torch.autocast('cuda',dtype=torch.float16):
            _,a,b,c=stages(model,normalize(x))
            features=torch.cat([F.adaptive_avg_pool2d(t,(4,4)).flatten(1) for t in [a,b,c]],1)
        output.append(features.float().cpu().numpy())
    return np.concatenate(output)


def baseline(args):
    setup(args.seed)
    data,pixels=load()
    feature_path=HERE/'frozen_features.npz'
    if not feature_path.exists():
        start=time.time()
        net=encoder().cuda().to(memory_format=torch.channels_last).eval()
        features=extract(net,pixels,False,args.batch_size)
        flipped=extract(net,pixels,True,args.batch_size)
        np.savez(feature_path,features=features,flipped=flipped)
        print('FEATURES',features.shape,'seconds',time.time()-start,flush=True)
    f=np.load(feature_path)
    features,flipped=f['features'],f['flipped']
    truth=data['target']
    points=data['target_input']/512
    flipped_points=points.copy()
    flipped_points[:,0]=(511/512)-points[:,0]
    all_results=[]
    constant=np.zeros((80,2))
    for fold in range(5):
        fit=np.flatnonzero(data['folds']!=fold)
        valid=np.flatnonzero(data['folds']==fold)
        constant[valid]=restore(np.tile(data['target_input'][fit].mean(0),(len(valid),1)),data,valid)
    print('CONSTANT_BASELINE_RMSE',rmse(constant,truth),flush=True)
    best=None
    for alpha in [10.,100.,1000.,10000.]:
        oof={False:np.zeros((80,2)),True:np.zeros((80,2))}
        test={False:[],True:[]}
        weights={}
        for fold in range(5):
            fit=np.flatnonzero(data['folds']!=fold)
            valid=np.flatnonzero(data['folds']==fold)
            assert not set(data['image_groups'][fit]) & set(data['image_groups'][valid])
            x=np.concatenate([features[fit],flipped[fit]])
            y=np.concatenate([points[fit],flipped_points[fit]])
            scaler=StandardScaler().fit(x)
            model=Ridge(alpha=alpha,solver='cholesky').fit(scaler.transform(x),y)
            weights.update({f'fold{fold}_mean':scaler.mean_,f'fold{fold}_scale':scaler.scale_,
                            f'fold{fold}_coef':model.coef_,f'fold{fold}_intercept':model.intercept_})
            for indices,is_test in [(valid,False),(np.arange(80,100),True)]:
                plain=model.predict(scaler.transform(features[indices]))*512
                other=model.predict(scaler.transform(flipped[indices]))*512
                other[:,0]=511-other[:,0]
                for tta,predicted in [(False,plain),(True,(plain+other)/2)]:
                    restored=restore(predicted,data,indices)
                    if is_test:
                        test[tta].append(restored)
                    else:
                        oof[tta][valid]=restored
        for tta in [False,True]:
            score=rmse(oof[tta],truth)
            result={'alpha':alpha,'flip_tta':tta,'oof_rmse':score,
                    'fold_rmse':[rmse(oof[tta][data['folds']==fold],truth[data['folds']==fold]) for fold in range(5)]}
            all_results.append(result)
            print('RIDGE',json.dumps(result),flush=True)
            if best is None or score<best['oof_rmse']:
                best=result
                np.save(HERE/'baseline_oof.npy',oof[tta])
                np.save(HERE/'baseline_test.npy',np.mean(test[tta],axis=0))
                np.savez(HERE/'baseline_weights.npz',**weights)
    report={'constant_oof_rmse':rmse(constant,truth),'best':best,'all_candidates':all_results,
            'description':'Frozen ImageNet ResNet18 spatial features with a fitted ridge coordinate-regression head; training includes horizontal-flip augmentation; decoded-image groups kept within folds.'}
    (HERE/'baseline_result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('BASELINE_DONE',json.dumps(best),flush=True)


def train(args):
    data,pixels=load()
    target=torch.from_numpy(data['target_input'].copy()).cuda()
    for fold in [int(v) for v in args.folds.split(',')]:
        setup(args.seed+fold*97)
        folder=HERE/f'heatmap_s{args.seed}_fold{fold}'
        folder.mkdir(exist_ok=True)
        if (folder/'result.json').exists():
            print('SKIP_COMPLETE_FOLD',fold,flush=True)
            continue
        model=LandmarkNet().cuda().to(memory_format=torch.channels_last)
        for p in model.enc.parameters():
            p.requires_grad_(False)
        ema=copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
        decoder=[p for name,p in model.named_parameters() if not name.startswith('enc.')]
        opt=torch.optim.AdamW([{'params':decoder,'lr':args.lr},{'params':model.enc.parameters(),'lr':0.}],weight_decay=1e-4)
        scaler=torch.amp.GradScaler('cuda')
        fit=np.flatnonzero(data['folds']!=fold)
        valid=np.flatnonzero(data['folds']==fold)
        assert not set(data['image_groups'][fit]) & set(data['image_groups'][valid])
        fit_gpu=torch.as_tensor(fit,device='cuda')
        history=[]
        best={'rmse':float('inf'),'epoch':0,'variant':None}
        t0=time.time()
        print('FOLD_START',fold,'train',len(fit),'validation',len(valid),'gpu',torch.cuda.get_device_name(0),flush=True)
        for epoch in range(1,args.epochs+1):
            start=time.time()
            if epoch==args.freeze_epochs+1:
                for p in model.enc.parameters():
                    p.requires_grad_(True)
            factor=.1+.9*(1+math.cos(math.pi*(epoch-1)/args.epochs))/2
            warmup=min(1.,epoch/3)
            opt.param_groups[0]['lr']=args.lr*factor*warmup
            opt.param_groups[1]['lr']=args.encoder_lr*factor if epoch>args.freeze_epochs else 0.
            model.train()
            order=fit_gpu[torch.randperm(len(fit_gpu),device='cuda')]
            losses=[]
            for begin in range(0,len(order),args.batch_size):
                index=order[begin:begin+args.batch_size]
                x,y=augment(pixels[index].float()/255,target[index])
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model(normalize(x))
                    loss=heatmap_loss(logits,y)
                assert bool(torch.isfinite(loss)), 'Non-finite training loss'
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(),5.)
                scaler.step(opt)
                scaler.update()
                losses.append(float(loss.detach()))
            update_ema(ema,model,.75)
            record={'epoch':epoch,'loss':float(np.mean(losses))}
            if epoch==1 or epoch%5==0 or epoch==args.epochs:
                for variant,network in [('raw',model),('ema',ema)]:
                    prediction=predict(network,pixels,valid,args.batch_size,False)['local']
                    score=rmse(restore(prediction,data,valid),data['target'][valid])
                    record['val_'+variant+'_rmse']=score
                    if score<best['rmse']:
                        best={'rmse':score,'epoch':epoch,'variant':variant}
                        torch.save({'state_dict':network.state_dict(),'best':best,'args':vars(args),'fold':fold},folder/'best.pt')
                record['best_rmse']=best['rmse']
            torch.cuda.synchronize()
            record['seconds']=time.time()-start
            record['peak_gpu_mb']=torch.cuda.max_memory_allocated()/1024**2
            history.append(record)
            (folder/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            if epoch==1 or epoch%5==0 or epoch==args.epochs:
                print('HEATMAP',fold,json.dumps(record),flush=True)
            if epoch>=60 and epoch-best['epoch']>=args.patience:
                print('EARLY_STOP',fold,json.dumps(best),flush=True)
                break
        checkpoint=torch.load(folder/'best.pt',weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        predictions={}
        scores={}
        for tta in [False,True]:
            for indices,part in [(valid,'valid'),(np.arange(80,100),'test')]:
                output=predict(model,pixels,indices,args.batch_size,tta)
                for kind,p in output.items():
                    key=f'{part}_{kind}_tta{int(tta)}'
                    predictions[key]=restore(p,data,indices)
                    if part=='valid':
                        scores[key]=rmse(predictions[key],data['target'][valid])
        np.savez(folder/'predictions.npz',valid_indices=valid,**predictions)
        report={'fold':fold,'seed':args.seed,'best':best,'scores':scores,'epochs_completed':len(history),
                'training_seconds':time.time()-t0,'args':vars(args),'gpu':torch.cuda.get_device_name(0)}
        (folder/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('FOLD_DONE',json.dumps(report),flush=True)
        del model,ema,opt,scaler
        torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['baseline','train'],required=True)
    p.add_argument('--folds',default='0,1,2,3,4')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--epochs',type=int,default=180)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--encoder-lr',type=float,default=2e-5)
    p.add_argument('--freeze-epochs',type=int,default=8)
    p.add_argument('--patience',type=int,default=50)
    args=p.parse_args()
    OUTPUT.mkdir(exist_ok=True,parents=True)
    if args.stage=='baseline':
        baseline(args)
    else:
        setup(args.seed)
        train(args)


if __name__=='__main__':
    main()
