"""Local CUDA training and template-aligned inference for Kaggle 2026task2."""
import argparse
import copy
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
WORK, OUT = ROOT, ROOT / 'outputs'


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.shortcut(x), inplace=True)


class FaceResNet(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        layers, channels = [], width
        for stage, count in enumerate([2, 2, 2, 2]):
            target = width * 2**stage
            for block in range(count):
                stride = 2 if stage and block == 0 else 1
                layers.append(Block(channels, target, stride))
                channels = target
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.15), nn.Linear(channels, 6))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def augment(x):
    b = x.shape[0]
    angle = (torch.rand(b, device=x.device) - .5) * (20 * math.pi/180)
    scale = .9 + .2 * torch.rand(b, device=x.device)
    flip = torch.where(torch.rand(b, device=x.device) < .5, -1., 1.)
    theta = torch.zeros((b, 2, 3), device=x.device)
    c, s = angle.cos()/scale, angle.sin()/scale
    theta[:, 0, 0], theta[:, 0, 1] = c*flip, -s
    theta[:, 1, 0], theta[:, 1, 1] = s*flip, c
    theta[:, :, 2] = (torch.rand(b, 2, device=x.device) - .5)*.12
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    contrast = .9 + .2 * torch.rand((b, 1, 1, 1), device=x.device)
    brightness = (torch.rand((b, 1, 1, 1), device=x.device)-.5)*.06
    means = x.mean((2, 3), keepdim=True)
    return ((x-means)*contrast + means + brightness).clamp(0, 1)


@torch.no_grad()
def predict(model, pixels, indices, mean, std, batch_size, tta=False):
    model.eval()
    result = []
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start+batch_size]
        x = pixels[ids].float().div_(255)
        x = (x-mean)/std
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda', dtype=torch.float16):
            p = model(x).float().softmax(1)
            if tta:
                p = (p + model(x.flip(-1)).float().softmax(1))/2
        result.append(p.cpu())
    return torch.cat(result).numpy()


@torch.no_grad()
def update_ema(ema, model, decay):
    for p, q in zip(ema.parameters(), model.parameters()):
        p.lerp_(q, 1-decay)
    for p, q in zip(ema.buffers(), model.buffers()):
        if p.dtype.is_floating_point:
            p.lerp_(q, 1-decay)
        else:
            p.copy_(q)


def make_model(width):
    return FaceResNet(width).cuda().to(memory_format=torch.channels_last)


def write_submission(ids, probs, filename):
    pred = probs.argmax(1).astype(int)
    assert probs.shape == (len(ids), 6) and np.isfinite(probs).all()
    assert len(ids) == len(set(ids.tolist())) == 7065
    assert set(pred.tolist()) <= set(range(6))
    destination = OUT / filename
    with destination.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Emotion'])
        writer.writerows(zip(ids.tolist(), pred.tolist()))
    with destination.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(ids)
    assert [r['ID'] for r in rows] == ids.tolist()
    assert all(0 <= int(r['Emotion']) <= 5 for r in rows)
    print('CSV_OK', str(destination), len(rows), dict(Counter(pred.tolist())), flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['validate', 'full'], default='validate')
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--schedule-epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=.08)
    parser.add_argument('--patience', type=int, default=18)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'CUDA GPU is required; CPU fallback is disabled.'
    OUT.mkdir(exist_ok=True)
    seed_everything(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cache = np.load(WORK / 'data_cache.npz')
    labels_np = cache['labels']
    train = torch.from_numpy(cache['train'].copy()).unsqueeze(1).cuda()
    test = torch.from_numpy(cache['test'].copy()).unsqueeze(1).cuda()
    labels = torch.from_numpy(labels_np.copy()).cuda()
    fit_np = cache['fit_idx'] if args.mode == 'validate' else np.arange(len(labels_np))
    fit = torch.from_numpy(fit_np.copy()).cuda()
    val_np = cache['val_idx']
    val = torch.from_numpy(val_np.copy()).cuda()
    sampled = train[fit].float()/255
    mean, std = float(sampled.mean()), float(sampled.std().clamp_min(.05))
    del sampled
    config = vars(args) | {'mean': mean, 'std': std, 'gpu': torch.cuda.get_device_name(0),
                          'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                          'fit_count': len(fit), 'validation_count': len(val) if args.mode == 'validate' else 0,
                          'architecture': '18-layer residual CNN, width 32, grayscale 48x48, small-image stem',
                          'label_order': ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']}
    (WORK / f'config_{args.mode}.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('CONFIG', json.dumps(config), flush=True)
    model = make_model(args.width)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9, weight_decay=5e-4, nesterov=True)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(label_smoothing=.05)
    history, best = [], {'accuracy': -1., 'epoch': 0, 'variant': None}
    total_start = time.time()
    log_path = WORK / f'history_{args.mode}.json'
    for epoch in range(1, args.epochs+1):
        epoch_start = time.time()
        warmup = min(1., epoch/4)
        schedule_epochs = args.schedule_epochs or args.epochs
        cosine = .5*(1+math.cos(math.pi*max(0, epoch-4)/max(1, schedule_epochs-4)))
        lr = args.lr*warmup*(.02+.98*cosine)
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        order = fit[torch.randperm(len(fit), device='cuda')]
        running_loss, correct, seen = 0., 0, 0
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start+args.batch_size]
            x = train[ids].float().div_(255)
            x = ((augment(x)-mean)/std).contiguous(memory_format=torch.channels_last)
            y = labels[ids]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema, model, .99 if epoch < 5 else .995)
            running_loss += float(loss.detach())*len(ids)
            correct += int((logits.detach().argmax(1)==y).sum())
            seen += len(ids)
        record = {'epoch': epoch, 'lr': lr, 'train_loss': running_loss/seen, 'train_accuracy': correct/seen}
        if args.mode == 'validate':
            for variant, network in [('raw', model), ('ema', ema)]:
                probs = predict(network, train, val, mean, std, args.batch_size)
                accuracy = float(np.mean(probs.argmax(1)==labels_np[val_np]))
                record['val_'+variant] = accuracy
                if accuracy > best['accuracy']:
                    best = {'accuracy': accuracy, 'epoch': epoch, 'variant': variant}
                    torch.save({'state_dict': network.state_dict(), 'config': config, 'best': best}, WORK/'best_validation.pt')
            record['best_accuracy'] = best['accuracy']
        else:
            if epoch == args.epochs or epoch % 10 == 0:
                torch.save({'state_dict': ema.state_dict(), 'config': config, 'epoch': epoch}, WORK/'full_model.pt')
        torch.cuda.synchronize()
        record['seconds'] = time.time()-epoch_start
        record['gpu_peak_mb'] = torch.cuda.max_memory_allocated()/1024**2
        history.append(record)
        log_path.write_text(json.dumps(history, indent=2), encoding='utf-8')
        print('EPOCH', json.dumps(record), flush=True)
        if args.mode == 'validate' and epoch >= 35 and epoch-best['epoch'] >= args.patience:
            print('EARLY_STOP', json.dumps(best), flush=True)
            break
    report = {'config': config, 'training_seconds': time.time()-total_start, 'epochs_completed': len(history)}
    if args.mode == 'validate':
        checkpoint = torch.load(WORK/'best_validation.pt', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        plain = predict(model, train, val, mean, std, args.batch_size, False)
        flipped = predict(model, train, val, mean, std, args.batch_size, True)
        plain_acc = float(np.mean(plain.argmax(1)==labels_np[val_np]))
        tta_acc = float(np.mean(flipped.argmax(1)==labels_np[val_np]))
        use_tta = tta_acc > plain_acc
        chosen = flipped if use_tta else plain
        matrix = np.zeros((6, 6), dtype=int)
        np.add.at(matrix, (labels_np[val_np], chosen.argmax(1)), 1)
        report.update({'best': checkpoint['best'], 'validation_accuracy': plain_acc,
                       'validation_flip_tta_accuracy': tta_acc, 'use_flip_tta': use_tta,
                       'confusion_matrix': matrix.tolist(),
                       'class_recall': (matrix.diagonal()/matrix.sum(1)).tolist()})
        probs = predict(model, test, torch.arange(len(test), device='cuda'), mean, std, args.batch_size, use_tta)
        write_submission(cache['test_ids'], probs, 'submission_validated.csv')
        np.save(WORK/'validation_model_test_probabilities.npy', probs)
        np.save(WORK/'validation_probabilities.npy', chosen)
    else:
        checkpoint = torch.load(WORK/'full_model.pt', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        val_report = json.loads((WORK/'result_validate.json').read_text(encoding='utf-8'))
        use_tta = bool(val_report['use_flip_tta'])
        probs = predict(model, test, torch.arange(len(test), device='cuda'), mean, std, args.batch_size, use_tta)
        write_submission(cache['test_ids'], probs, 'submission.csv')
        np.save(WORK/'full_model_test_probabilities.npy', probs)
        report.update({'use_flip_tta': use_tta, 'validation_accuracy_of_selection_run': val_report['validation_accuracy'],
                       'note': 'Final model retrained on all training labels. Selection-run validation accuracy is not a final-model or test score.'})
    (WORK/f'result_{args.mode}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('RESULT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
