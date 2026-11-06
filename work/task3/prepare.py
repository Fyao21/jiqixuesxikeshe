import csv
import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import StratifiedGroupKFold
from scipy import ndimage

HERE = Path(__file__).resolve().parent
SOURCE = Path(r'C:\Users\w2922\Downloads\2026task3.zip')
SIDE = 512


def read_csv(z, name):
    return list(csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig'), newline='')))


def main():
    global SOURCE
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',default=str(SOURCE))
    SOURCE=Path(parser.parse_args().archive)
    HERE.mkdir(exist_ok=True)
    original_dir = HERE / 'originals'
    original_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(SOURCE) as z:
        rows = read_csv(z, 'detection/fovea_localization_train_GT.csv')
        template = read_csv(z, 'detection/sample_submission.csv')
        ids = np.asarray([int(r['data']) for r in rows], dtype=np.int64)
        assert len(ids) == len(set(ids.tolist())) == 80
        target = np.asarray([[float(r['Fovea_X']), float(r['Fovea_Y'])] for r in rows], dtype=np.float32)
        test_ids = sorted(set(int(r['ImageID'].split('_')[0]) for r in template))
        assert len(test_ids) == 20 and len(template) == 40
        for i in test_ids:
            assert {r['ImageID'] for r in template if int(r['ImageID'].split('_')[0]) == i} == {f'{i}_Fovea_X', f'{i}_Fovea_Y'}
        all_ids = ids.tolist() + test_ids
        pixels, sizes, scales, pads, crops, digests, boxes, observations = [], [], [], [], [], [], [], []
        thumbs = Image.new('RGB', (5*260, 4*210), 'white')
        draw = ImageDraw.Draw(thumbs)
        for index, image_id in enumerate(all_ids):
            part = 'train' if index < len(ids) else 'test'
            name = f'detection/{part}/{image_id:04d}.jpg'
            payload = z.read(name)
            (original_dir / f'{image_id:04d}.jpg').write_bytes(payload)
            with Image.open(io.BytesIO(payload)) as im:
                im = im.convert('RGB')
                w, h = im.size
                small = np.asarray(im.resize((384,384),Image.Resampling.BILINEAR))
                mask = small.max(2) > 18
                components, count = ndimage.label(mask)
                areas = np.bincount(components.ravel())
                areas[0] = 0
                largest = components == areas.argmax()
                yy,xx = np.nonzero(largest)
                assert len(xx) > 384*384*.1, (image_id,'FOV too small')
                x0,y0 = max(0,int(xx.min()/384*w)-20),max(0,int(yy.min()/384*h)-20)
                x1,y1 = min(w,int((xx.max()+1)/384*w)+20),min(h,int((yy.max()+1)/384*h)+20)
                cw,ch=x1-x0,y1-y0
                s = min(SIDE/cw, SIDE/ch)
                rw, rh = round(cw*s), round(ch*s)
                px, py = (SIDE-rw)//2, (SIDE-rh)//2
                resized = im.crop((x0,y0,x1,y1)).resize((rw, rh), Image.Resampling.BILINEAR)
                canvas = Image.new('RGB', (SIDE, SIDE))
                canvas.paste(resized, (px, py))
                pixels.append(np.asarray(canvas).copy())
                sizes.append([w, h])
                # Point coordinates follow PIL's pixel-center resize convention.
                scales.append([rw/cw, rh/ch])
                pads.append([px, py])
                crops.append([x0,y0,x1,y1])
                digests.append(hashlib.sha256(im.tobytes()).hexdigest())
                if index < len(ids):
                    xy = target[index]
                    assert (xy == 0).all() or (0 <= xy[0] < w and 0 <= xy[1] < h), (image_id, xy, (w,h))
                    ann = ET.fromstring(z.read(f'detection/train_location/{image_id:04d}.xml'))
                    xml_size = [int(ann.findtext('size/width')), int(ann.findtext('size/height'))]
                    assert xml_size == [w,h], (image_id, xml_size, (w,h))
                    found = []
                    for obj in ann.findall('object'):
                        if obj.findtext('name').lower() == 'fovea':
                            found.append([float(obj.findtext('bndbox/'+k)) for k in ['xmin','ymin','xmax','ymax']])
                    assert len(found) == 1
                    box = found[0]
                    boxes.append(box)
                    if (xy != 0).any():
                        assert box[0] <= xy[0] <= box[2] and box[1] <= xy[1] <= box[3], (image_id, xy, box)
                    observations.append({'id': image_id, 'width': w, 'height': h, 'point': xy.tolist(), 'box': box})
                    if index < 20:
                        thumb = canvas.resize((200,200))
                        ox, oy = (index%5)*260, (index//5)*210
                        thumbs.paste(thumb, (ox+55,oy))
                        point = ((xy-np.asarray([x0,y0])+.5)*np.asarray([rw/cw,rh/ch])-.5+np.asarray([px,py]))*(200/SIDE)
                        tx, ty = ox+55+float(point[0]), oy+float(point[1])
                        draw.line((tx-5,ty,tx+5,ty), fill='lime', width=1)
                        draw.line((tx,ty-5,tx,ty+5), fill='lime', width=1)
                        draw.text((ox+3,oy+90), str(image_id), fill='black')
        imgs, sizes, scales, pads, crops = np.stack(pixels), np.asarray(sizes), np.asarray(scales), np.asarray(pads),np.asarray(crops)
        train_target = (target-crops[:80,:2]+.5)*scales[:80]-.5+pads[:80]
        visible = np.any(target != 0, axis=1)
        train_target[~visible] = 0
        restored = (train_target-pads[:80]+.5)/scales[:80]-.5+crops[:80,:2]
        assert np.max(np.abs(restored[visible]-target[visible])) < 1e-3
        _, image_groups = np.unique(np.asarray(digests), return_inverse=True)
        duplicate_groups = []
        for group in np.unique(image_groups):
            members = np.flatnonzero(image_groups == group)
            if len(members) > 1:
                duplicate_groups.append([all_ids[i] for i in members])
        # No patient identifiers are supplied. Stratify only by acquisition size.
        # Labels and annotated bounding boxes are not used to construct folds.
        size_labels = np.asarray([f'{w}x{h}' for w,h in sizes[:80]])
        counts = Counter(size_labels.tolist())
        strata = np.asarray([x if counts[x] >= 5 else 'rare' for x in size_labels])
        if min(Counter(strata.tolist()).values()) < 5:
            # Rare image sizes are distributed with the common acquisition group.
            strata[np.asarray([counts[x]<5 for x in size_labels])] = Counter(strata.tolist()).most_common(1)[0][0]
        folds = np.empty(80, dtype=np.int64)
        for fold, (fit, valid) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=2026).split(np.zeros(80), strata, image_groups[:80])):
            folds[valid] = fold
            assert not set(image_groups[fit]) & set(image_groups[valid])
        np.savez(HERE/'data.npz', images=imgs, ids=np.asarray(all_ids), sizes=sizes, scales=scales, pads=pads,crops=crops,
                 target=target, target_input=train_target.astype(np.float32), visible=visible, boxes=np.asarray(boxes), folds=folds,
                 template_ids=np.asarray([r['ImageID'] for r in template]), image_groups=image_groups)
        report = {'archive': str(SOURCE), 'archive_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  'train_count':80, 'test_count':20, 'submission_rows':40,
                  'all_size_counts':dict(Counter(f'{w}x{h}' for w,h in sizes)),
                  'invisible_training_count':int((~visible).sum()), 'unique_images':len(set(digests)),
                  'duplicate_groups':duplicate_groups,
                  'train_test_overlap_groups':[g for g in duplicate_groups if min(g)<=80<max(g)],
                  'point_box_max_center_distance':float(np.max(np.linalg.norm(target-(np.asarray(boxes)[:,:2]+np.asarray(boxes)[:,2:])/2,axis=1))),
                  'folds':folds.tolist(), 'split_note':'Five acquisition-size-stratified group folds, seed 2026; pixel-identical images kept together. No patient identifiers supplied.',
                  'point_resize_convention':'x_input=(x_original-crop_x0+0.5)*scale_x-0.5+pad_x, and same for y',
                  'fov_crop':'Largest RGB max>18 connected component at 384x384, 20 original-pixel margin, image-only preprocessing',
                  'observations':observations}
        (HERE/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        thumbs.save(HERE/'training_points.jpg')
        print(json.dumps({k:v for k,v in report.items() if k!='observations'},ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
