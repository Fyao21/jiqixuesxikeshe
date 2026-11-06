import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from model import HERE, VesselNet, augment, loss_fn, predict


def dice_scores(probs, targets, threshold=.5):
    p = probs >= threshold
    return (2 * (p * targets).sum((1, 2)) + 1e-7) / (p.sum((1, 2)) + targets.sum((1, 2)) + 1e-7)


def train_fold(args, data, fold):
    folder = HERE / f'fold{fold}'
    folder.mkdir(exist_ok=True)
    if (folder / 'result.json').exists():
        print('SKIP completed fold', fold, flush=True)
        return
    seed = 2026 + 97 * fold
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    train_idx = np.flatnonzero(data['folds'] != fold)
    valid_idx = np.flatnonzero(data['folds'] == fold)
    x = torch.from_numpy(data['images'][:20].transpose(0, 3, 1, 2).copy()).to('cuda')
    y = torch.from_numpy(data['masks'][:, None].astype('float32')).to('cuda')
    model = VesselNet(pretrained=True).to('cuda', memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    enc = list(model.enc.parameters())
    enc_ids = {id(p) for p in enc}
    decoder = [p for p in model.parameters() if id(p) not in enc_ids]
    optimizer = torch.optim.AdamW([{'params': enc, 'lr': .0001}, {'params': decoder, 'lr': .001}], weight_decay=.0001)
    scaler = torch.amp.GradScaler('cuda')
    best, history = {'dice': -1., 'epoch': 0}, []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        rate = min(1., epoch / 5) * (.1 + .9 * .5 * (1 + math.cos(math.pi * (epoch - 1) / args.epochs)))
        optimizer.param_groups[0]['lr'] = .0001 * rate
        optimizer.param_groups[1]['lr'] = .001 * rate
        order = np.random.permutation(train_idx)
        losses = []
        for start_i in range(0, len(order), args.batch_size):
            indices = order[start_i:start_i + args.batch_size]
            a, b = augment(x[indices].float() / 255, y[indices])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(a)
                loss = loss_fn(logits, b)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer); scaler.update()
            with torch.no_grad():
                for dst, src in zip(ema.parameters(), model.parameters()):
                    dst.lerp_(src, .05)
                for dst, src in zip(ema.buffers(), model.buffers()):
                    dst.copy_(src)
            losses.append(loss.item())
        row = dict(epoch=epoch, loss=float(np.mean(losses)), seconds=time.perf_counter() - start)
        if epoch % 10 == 0:
            probs = predict(ema, x[valid_idx], batch_size=2, transforms=1, amp=True)
            score = float(dice_scores(probs, data['masks'][valid_idx]).mean())
            row['valid_dice_at_0.5'] = score
            if score > best['dice']:
                best = dict(dice=score, epoch=epoch)
                state = {k: v.detach().cpu() for k, v in ema.state_dict().items()}
                torch.save(dict(state_dict=state, fold=fold, seed=seed, best=best, args=vars(args),
                                train_ids=(train_idx + 1).tolist(), valid_ids=(valid_idx + 1).tolist()), folder / 'best.pt')
            print(json.dumps(dict(fold=fold, **row, best=best)), flush=True)
            (folder / 'history.json').write_text(json.dumps(history + [row], indent=2), encoding='utf-8')
        history.append(row)
        if epoch >= 120 and epoch - best['epoch'] >= 60:
            break
    torch.cuda.synchronize()
    result = dict(fold=fold, best=best, epochs_trained=epoch, seconds=time.perf_counter() - start,
                  gpu=torch.cuda.get_device_name(), seed=seed, validation_mode='AMP EMA single view at threshold 0.5')
    (folder / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    (folder / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('COMPLETED', json.dumps(result), flush=True)
    del model, ema, x, y
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', default='0,1,2,3,4')
    parser.add_argument('--epochs', type=int, default=220)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = True
    print('ENV', torch.__version__, torch.cuda.get_device_name(), flush=True)
    data = np.load(HERE / 'data.npz')
    for fold in map(int, args.folds.split(',')):
        train_fold(args, data, fold)
