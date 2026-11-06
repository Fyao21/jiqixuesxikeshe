import csv
import hashlib
import json
import ast
import zipfile
from pathlib import Path

import numpy as np
import torchvision
from PIL import Image, ImageDraw

from model import HERE
from prepare import OUT
from train import dice_scores


def encode(mask):
    flat = mask.reshape(-1).astype(np.int8)
    changes = np.flatnonzero(np.diff(np.r_[0, flat, 0]))
    starts, lengths = changes[::2] + 1, changes[1::2] - changes[::2]
    return ' '.join(map(str, np.c_[starts, lengths].reshape(-1)))


def decode(rle):
    pairs = np.array([int(v) for v in rle.split()], dtype=np.int64).reshape(-1, 2)
    flat = np.zeros(512 * 512, np.uint8)
    if len(pairs):
        assert (pairs > 0).all()
        assert (pairs[:, 0] + pairs[:, 1] - 1 <= len(flat)).all()
        assert (pairs[1:, 0] >= pairs[:-1, 0] + pairs[:-1, 1]).all()
    for start, length in pairs:
        flat[start - 1:start - 1 + length] = 1
    return flat.reshape(512, 512)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(HERE / 'data.npz')
    targets = data['masks']
    assembled = {}
    reports = []
    for view in [1, 4, 8]:
        oof = np.zeros((20, 512, 512), np.float32)
        test = np.zeros_like(oof)
        seen = np.zeros(20, np.int64)
        for fold in range(5):
            p = np.load(HERE / f'fold{fold}' / 'predictions.npz')
            indices = p['indices']
            valid = indices[indices < 20]
            assert (data['folds'][valid] == fold).all()
            probs = p[f'view{view}']
            oof[valid] = probs[:len(valid)]
            seen[valid] += 1
            test += probs[len(valid):] / 5
        assert (seen == 1).all()
        assembled[view] = (oof, test)
        for threshold in np.round(np.arange(.25, .751, .025), 3):
            scores = dice_scores(oof, targets, threshold)
            reports.append(dict(views=view, threshold=float(threshold), mean_dice=float(scores.mean()),
                                fold_dice=[float(scores[data['folds'] == f].mean()) for f in range(5)]))
    best = max(reports, key=lambda x: x['mean_dice'])
    view, threshold = best['views'], best['threshold']
    oof, test = assembled[view]
    masks = test >= threshold
    path = OUT / 'submission_task4.csv'
    pngdir = HERE / 'test_masks'
    pngdir.mkdir(exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Predicted'])
        for i, mask in enumerate(masks, 1):
            rle = encode(mask)
            assert np.array_equal(decode(rle), mask)
            writer.writerow([i, rle])
            Image.fromarray(mask.astype(np.uint8) * 255).save(pngdir / f'{i}.png')
    # Independent implementation supplied inside the original dataset.
    # Execute the exact supplied conversion function; its unrelated CSV-writing
    # helper imports pandas, which is unnecessary for this independent check.
    syntax = ast.parse((HERE / 'official_segmentation_to_csv.py').read_text(encoding='utf-8-sig'))
    function = next(n for n in syntax.body if isinstance(n, ast.FunctionDef) and n.name == 'turn_to_str')
    official = dict(Image=Image, np=np, torchvision=torchvision)
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'official_segmentation_to_csv.py', 'exec'), official)
    official_rles = official['turn_to_str']([str(pngdir / f'{i}.png') for i in range(1, 21)])
    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ['Id', 'Predicted']
        rows = list(reader)
    assert [int(r['Id']) for r in rows] == data['ids'].tolist()
    assert [r['Predicted'] for r in rows] == official_rles
    assert all(np.array_equal(decode(r['Predicted']), masks[i]) for i, r in enumerate(rows))
    with zipfile.ZipFile(OUT / '测试集分割掩膜.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for i in range(1, 21):
            z.write(pngdir / f'{i}.png', f'{i}.png')
    scores = dice_scores(oof, targets, threshold)
    with (OUT / '折外验证逐图指标.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(['Id', 'fold', 'dice', 'threshold', 'views'])
        for i, score in enumerate(scores):
            w.writerow([i + 1, int(data['folds'][i]), f'{score:.8f}', threshold, view])
    canvas = Image.new('RGB', (3 * 384, 8 * 280), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, idx in enumerate(np.argsort(scores)[:8]):
        image = Image.fromarray(data['images'][idx]).resize((256, 256))
        gt = Image.fromarray(targets[idx] * 255).convert('RGB').resize((256, 256), Image.Resampling.NEAREST)
        pr = Image.fromarray((oof[idx] >= threshold).astype(np.uint8) * 255).convert('RGB').resize((256, 256), Image.Resampling.NEAREST)
        for col, im in enumerate([image, gt, pr]):
            canvas.paste(im, (col * 384, row * 280 + 24))
        draw.text((0, row * 280 + 4), f'TRAIN OOF ID {idx+1}, fold {data["folds"][idx]}', fill='black')
        draw.text((384, row * 280 + 4), 'Ground truth: official resize', fill='black')
        draw.text((768, row * 280 + 4), f'Prediction, Dice={scores[idx]:.4f}', fill='black')
    canvas.save(OUT / '折外分割效果检查.png')
    testcanvas = Image.new('RGB', (4 * 256, 5 * 280), 'white')
    draw = ImageDraw.Draw(testcanvas)
    for i in range(20):
        image = data['images'][20 + i].astype(np.float32)
        overlay = image.copy()
        overlay[masks[i]] = .45 * image[masks[i]] + .55 * np.array([0, 255, 0])
        row, col = divmod(i, 4)
        testcanvas.paste(Image.fromarray(overlay.astype(np.uint8)).resize((256, 256)), (col * 256, row * 280 + 24))
        draw.text((col * 256 + 4, row * 280 + 5), f'TEST {i+1}: predicted vessels', fill='black')
    testcanvas.save(OUT / '测试集预测预览.png')
    audit = json.loads((HERE / 'audit.json').read_text(encoding='utf-8'))
    result = dict(task='2026task-4 retinal vessel segmentation', selected=best,
                  fixed_threshold_0_5=[r for r in reports if r['threshold'] == .5], candidates=reports,
                  oof_image_dice=scores.tolist(), test_vessel_fractions=masks.mean((1, 2)).tolist(),
                  model='ImageNet ResNet18 U-Net, full resolution skip; five fold EMA ensemble',
                  training=[json.loads((HERE / f'fold{f}' / 'result.json').read_text()) for f in range(5)],
                  inference=json.loads((HERE / 'inference_check.json').read_text()),
                  submission=dict(path=str(path.resolve()), rows=20, columns=['Id', 'Predicted'],
                                  mask_shape=[512, 512], row_major=True, one_based=True, round_trip_exact=True,
                                  official_encoder_matches=True, sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                  source_archive_sha256=audit['archive_sha256'],
                  notes=['Checkpoint selection, threshold and TTA selection use validation data. OOF scores may be optimistic; not Kaggle test scores.',
                         'Actual supplied labels are white vessels; official encoder uses nonzero as foreground, despite conflicting description prose.',
                         'All images are split as whole images; no duplicate images; patient identifiers unavailable.',
                         'No external retinal images or test labels used. ImageNet initialization only.',
                         'No Kaggle submission was made.'])
    (OUT / '训练与提交检查.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(HERE / 'final_probabilities.npz', oof=oof, test=test, threshold=threshold, views=view)
    print(json.dumps(dict(selected=best, fixed=result['fixed_threshold_0_5'], submission=result['submission']), indent=2), flush=True)


if __name__ == '__main__':
    main()
