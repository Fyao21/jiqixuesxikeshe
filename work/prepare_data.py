import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / 'work'
SOURCE = Path(r'C:\Users\w2922\Downloads\2026task2.zip')
LABELS = {'angry': 0, 'anger': 0, 'fear': 1, 'fearful': 1,
          'happy': 2, 'happiness': 2, 'sad': 3, 'sadness': 3,
          'surprise': 4, 'surprised': 4, 'neutral': 5}


def main():
    WORK.mkdir(exist_ok=True)
    with zipfile.ZipFile(SOURCE) as archive:
        entries = [n for n in archive.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
        train_names = sorted(n for n in entries if '/train/' in n)
        test_names = sorted(n for n in entries if '/test/' in n)
        rows = list(csv.DictReader(io.StringIO(archive.read('submission.csv').decode('utf-8-sig'), newline='')))
        assert rows and list(rows[0]) == ['ID', 'Emotion'], 'Unexpected template columns'
        ids = [r['ID'] for r in rows]
        by_id = {Path(n).name: n for n in test_names}
        assert len(by_id) == len(test_names) == len(ids) == len(set(ids))
        assert set(ids) == set(by_id), 'Template and test IDs do not match'
        test_names = [by_id[i] for i in ids]
        arrays, targets, digests, modes, sizes = [], [], [], Counter(), Counter()
        for kind, names in [('train', train_names), ('test', test_names)]:
            images = []
            for i, name in enumerate(names):
                with Image.open(io.BytesIO(archive.read(name))) as im:
                    modes[im.mode] += 1
                    sizes[str(im.size)] += 1
                    assert im.size == (48, 48), (name, im.size)
                    pixels = np.asarray(im.convert('L'), dtype=np.uint8).copy()
                images.append(pixels)
                if kind == 'train':
                    folder = name.split('/')[-2].lower()
                    assert folder in LABELS, folder
                    targets.append(LABELS[folder])
                    digests.append(hashlib.sha256(pixels.tobytes()).hexdigest())
                if (i + 1) % 5000 == 0:
                    print(f'DECODE {kind} {i+1}/{len(names)}', flush=True)
            arrays.append(np.stack(images))
        labels = np.asarray(targets, dtype=np.int64)
        _, groups = np.unique(np.asarray(digests), return_inverse=True)
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels, groups))
        assert not set(groups[train_idx]) & set(groups[val_idx])
        assert set(labels[train_idx]) == set(labels[val_idx]) == set(range(6))
        conflicts = defaultdict(set)
        for group, label in zip(groups, labels):
            conflicts[int(group)].add(int(label))
        report = {
            'source_archive': str(SOURCE),
            'train_count': len(train_names), 'test_count': len(test_names),
            'label_mapping': {'Angry': 0, 'Fear': 1, 'Happy': 2, 'Sad': 3, 'Surprise': 4, 'Neutral': 5},
            'class_counts': {str(k): int(v) for k,v in sorted(Counter(targets).items())},
            'image_modes': dict(modes), 'image_sizes': dict(sizes),
            'duplicate_train_images_beyond_first': len(labels)-len(np.unique(groups)),
            'duplicate_groups_with_conflicting_labels': sum(len(v)>1 for v in conflicts.values()),
            'validation_split': 'StratifiedGroupKFold, five folds, first fold, seed 42; pixel-identical images kept together',
            'fit_count': len(train_idx), 'validation_count': len(val_idx),
            'fit_class_counts': {str(k): int(v) for k,v in sorted(Counter(labels[train_idx].tolist()).items())},
            'validation_class_counts': {str(k): int(v) for k,v in sorted(Counter(labels[val_idx].tolist()).items())},
            'csv_columns': ['ID', 'Emotion'],
        }
        np.savez(WORK / 'data_cache.npz', train=arrays[0], test=arrays[1], labels=labels,
                 fit_idx=train_idx, val_idx=val_idx, test_ids=np.asarray(ids), train_paths=np.asarray(train_names))
        (WORK / 'data_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
