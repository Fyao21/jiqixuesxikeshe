import argparse
import hashlib
import json

import numpy as np
import torch

from model import HERE, VesselNet, predict


def main(args):
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    data = np.load(HERE / 'data.npz')
    images = torch.from_numpy(data['images'].transpose(0, 3, 1, 2).copy()).to('cuda')
    checks = []
    for fold in map(int, args.folds.split(',')):
        folder = HERE / f'fold{fold}'
        checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        model = VesselNet().to('cuda', memory_format=torch.channels_last)
        model.load_state_dict(checkpoint['state_dict'])
        valid_idx = np.flatnonzero(data['folds'] == fold)
        indices = np.r_[valid_idx, np.arange(20, 40)]
        target = folder / 'predictions.npz'
        old = np.load(target) if target.exists() else None
        hash_path = folder / 'prediction_hashes.json'
        reference_hashes = json.loads(hash_path.read_text()) if hash_path.exists() else {}
        current_hashes = {}
        predictions = dict(indices=indices)
        comparison = []
        for tta in map(int, args.views.split(',')):
            key = f'view{tta}'
            probs = predict(model, images[indices], batch_size=2, transforms=tta, amp=False)
            assert np.isfinite(probs).all() and probs.min() >= 0 and probs.max() <= 1
            predictions[key] = probs
            current_hashes[key] = hashlib.sha256(probs.tobytes()).hexdigest()
            if old is not None and key in old.files:
                difference = float(np.max(np.abs(probs - old[key])))
                assert difference == 0, (fold, key, difference)
                comparison.append(dict(view=tta, max_probability_difference=difference))
            elif key in reference_hashes:
                assert current_hashes[key] == reference_hashes[key], (fold, key, 'reference probability hash differs')
                comparison.append(dict(view=tta, probability_sha256_matches=True))
            print('INFER', fold, key, 'shape', probs.shape, 'range', float(probs.min()), float(probs.max()), flush=True)
        if old is not None:
            old.close()
        temporary = folder / 'predictions.pending.npz'
        np.savez_compressed(temporary, **predictions)
        temporary.replace(target)
        hash_path.write_text(json.dumps(current_hashes, indent=2), encoding='utf-8')
        checks.append(dict(fold=fold, compared_to_saved=len(comparison) == len(current_hashes), comparisons=comparison,
                           checkpoint_sha256=hashlib.sha256((folder / 'best.pt').read_bytes()).hexdigest()))
        del model
        torch.cuda.empty_cache()
    report = dict(dtype='float32', deterministic_cudnn=True, tf32=False,
                  independent_rerun_matches=bool(checks) and all(c['compared_to_saved'] for c in checks), folds=checks)
    (HERE / ('inference_check.json' if args.folds == '0,1,2,3,4' else 'inference_partial.json')).write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--folds', default='0,1,2,3,4')
    p.add_argument('--views', default='1,4,8')
    main(p.parse_args())
