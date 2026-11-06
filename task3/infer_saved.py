"""Recompute all held-out/test predictions from saved weights, without training."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import torch

from models import LandmarkNet,predict,restore
from run import setup,load

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--accept-new-precision',action='store_true')
    args=parser.parse_args()
    setup(2026)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    data,pixels=load()
    checks=[]
    for fold in range(5):
        folder=HERE/f'heatmap_s2026_fold{fold}'
        checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        assert checkpoint['fold']==fold
        model=LandmarkNet(pretrained=False).cuda().to(memory_format=torch.channels_last)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        valid=np.flatnonzero(data['folds']==fold)
        fit=np.flatnonzero(data['folds']!=fold)
        assert not set(data['image_groups'][valid]) & set(data['image_groups'][fit])
        results={}
        for tta in [0,1]:
            for part,indices in [('valid',valid),('test',np.arange(80,100))]:
                output=predict(model,pixels,indices,8,bool(tta),amp=False)
                for kind,p in output.items():
                    results[f'{part}_{kind}_tta{tta}']=restore(p,data,indices)
        previous=folder/'predictions.npz'
        difference=None
        if previous.exists():
            old=np.load(previous)
            difference=max(float(np.max(np.abs(results[k]-old[k]))) for k in results)
            if args.accept_new_precision:
                shutil.copy2(previous,folder/'predictions_amp_reference.npz')
            else:
                assert difference<.01, ('Inference reproduction differs',fold,difference)
        np.savez(previous,valid_indices=valid,**results)
        print('INFERENCE_REPRODUCED',fold,'max_difference_pixels',difference,flush=True)
        checks.append({'fold':fold,'max_difference_original_pixels':difference})
        del model
        torch.cuda.empty_cache()
    report={'precision':'float32','cudnn_deterministic':True,'cudnn_benchmark':False,'tf32':False,
            'comparison':'previous AMP outputs' if args.accept_new_precision else 'independent repeat of deterministic FP32 outputs',
            'independent_reproduction_passed':not args.accept_new_precision and all(c['max_difference_original_pixels'] is not None for c in checks),'folds':checks}
    (HERE/'inference_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
