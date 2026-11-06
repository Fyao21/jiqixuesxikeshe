import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'


def prepare(archive):
    OUT.mkdir(parents=True, exist_ok=True)
    images, masks, native, hashes = [], [], [], []
    with zipfile.ZipFile(archive) as z:
        for split in ['train', 'test']:
            for i in range(1, 21):
                im = Image.open(io.BytesIO(z.read(f'segmentation/{split}/image/{i}.jpg'))).convert('RGB')
                assert im.size == (565, 584)
                hashes.append(hashlib.sha256(np.array(im).tobytes()).hexdigest())
                images.append(np.array(im.resize((512, 512), Image.Resampling.BILINEAR)))
                if split == 'train':
                    m = Image.open(io.BytesIO(z.read(f'segmentation/train/label/{i}.jpg'))).convert('L')
                    assert m.size == im.size
                    assert set(np.unique(np.array(m))) == {0, 255}
                    native.append(np.array(m))
                    # Exactly the supplied encoder's resize and nonzero convention.
                    masks.append(np.array(m.resize((512, 512), Image.Resampling.BILINEAR)) > 0)
        ids = [int(r['Id']) for r in csv.DictReader(io.StringIO(z.read('segmentation/sample_submission.csv').decode('utf-8-sig')))]
        assert ids == list(range(1, 21))
        (HERE / 'official_segmentation_to_csv.py').write_bytes(z.read('segmentation/segmentation_to_csv.py'))
    assert len(set(hashes)) == 40, 'Duplicate images: use grouped folds instead'
    folds = np.zeros(20, np.int64)
    for f, (_, valid) in enumerate(KFold(5, shuffle=True, random_state=2026).split(images[:20])):
        folds[valid] = f
    np.savez_compressed(HERE / 'data.npz', images=np.array(images), masks=np.array(masks, np.uint8),
                        native_masks=np.array(native), folds=folds, ids=ids)
    audit = dict(archive=str(Path(archive).resolve()), archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
                 train_images=20, test_images=20, original_size_wh=[565, 584], encoded_size_wh=[512, 512],
                 image_sha256=hashes, all_images_unique=True, folds=folds.tolist(),
                 native_vessel_fraction=(np.array(native) > 0).mean((1, 2)).tolist(),
                 encoded_vessel_fraction=np.array(masks).mean((1, 2)).tolist(),
                 target_transform='PIL bilinear resize to 512x512 then >0, as official encoder',
                 polarity='255/nonzero is vessel in actual supplied labels and encoder; webpage prose disagrees')
    (HERE / 'audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--archive', default=r'C:\Users\w2922\Downloads\2026task-4.zip')
    prepare(p.parse_args().archive)
