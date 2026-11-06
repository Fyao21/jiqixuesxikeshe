import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image,ImageDraw

from models import rmse

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    data=np.load(HERE/'data.npz')
    target=data['target']
    candidates={}
    for kind in ['local','global']:
        for tta in [0,1]:
            oof=np.full((80,2),np.nan)
            testing=[]
            for fold in range(5):
                folder=HERE/f'heatmap_s2026_fold{fold}'
                assert (folder/'result.json').exists(),f'Incomplete fold {fold}'
                p=np.load(folder/'predictions.npz')
                indices=p['valid_indices']
                assert np.array_equal(indices,np.flatnonzero(data['folds']==fold))
                oof[indices]=p[f'valid_{kind}_tta{tta}']
                testing.append(p[f'test_{kind}_tta{tta}'])
            assert np.isfinite(oof).all()
            name=f'heatmap_{kind}_flip{tta}'
            candidates[name]={'oof':oof,'test':np.mean(testing,0),'rmse':rmse(oof,target)}
    ridge={'oof':np.load(HERE/'baseline_oof.npy'),'test':np.load(HERE/'baseline_test.npy')}
    ridge['rmse']=rmse(ridge['oof'],target)
    candidates['frozen_feature_ridge']=ridge
    # Each blended OOF row contains predictions only from models that did not
    # train on that row. Test predictions then average all five fold models.
    pure_heatmap=min([n for n in candidates if n.startswith('heatmap')],key=lambda n:candidates[n]['rmse'])
    for weight in [.25,.5,.75]:
        n=f'heatmap_ridge_mix_{weight}'
        oof=weight*candidates[pure_heatmap]['oof']+(1-weight)*ridge['oof']
        test=weight*candidates[pure_heatmap]['test']+(1-weight)*ridge['test']
        candidates[n]={'oof':oof,'test':test,'rmse':rmse(oof,target)}
    best=min(candidates,key=lambda n:candidates[n]['rmse'])
    # Require a measurable improvement to add another model family.
    if best.startswith('heatmap_ridge_mix') and candidates[best]['rmse']>.99*candidates[pure_heatmap]['rmse']:
        best=pure_heatmap
    chosen=candidates[best]
    test_pred=chosen['test'].copy()
    assert test_pred.shape==(20,2) and np.isfinite(test_pred).all()
    unclipped=test_pred.copy()
    test_pred=np.maximum(test_pred,0)
    test_pred=np.minimum(test_pred,data['sizes'][80:]-1)
    clipped_rows=int(np.any(test_pred!=unclipped,axis=1).sum())
    audit=json.loads((HERE/'audit.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(audit['archive']) as z:
        template=list(csv.DictReader(io.StringIO(z.read('detection/sample_submission.csv').decode('utf-8-sig'),newline='')))
        expected_test={int(Path(n).stem) for n in z.namelist() if n.startswith('detection/test/') and n.endswith('.jpg')}
    ids=data['ids'][80:]
    assert set(ids.tolist())==expected_test
    predicted={int(i):v for i,v in zip(ids,test_pred)}
    rows=[]
    for row in template:
        image_id,axis=row['ImageID'].split('_Fovea_')
        assert axis in ['X','Y']
        rows.append({'ImageID':row['ImageID'],'value':f'{predicted[int(image_id)][0 if axis=="X" else 1]:.6f}'})
    path=OUT/'submission_task3.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['ImageID','value'])
        writer.writeheader()
        writer.writerows(rows)
    with path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==['ImageID','value']
        written=list(reader)
    assert len(written)==len(template)==40
    assert len({r['ImageID'] for r in written})==40
    assert [r['ImageID'] for r in written]==[r['ImageID'] for r in template]
    for row in written:
        i,axis=row['ImageID'].split('_Fovea_')
        value=float(row['value'])
        position=np.flatnonzero(ids==int(i))[0]+80
        assert np.isfinite(value) and 0<=value<=data['sizes'][position,0 if axis=='X' else 1]-1
    fold_reports=[json.loads((HERE/f'heatmap_s2026_fold{fold}'/'result.json').read_text(encoding='utf-8')) for fold in range(5)]
    report={
        'selected_model':best,'oof_rmse_pixels':chosen['rmse'],
        'candidate_oof_rmse_pixels':{k:v['rmse'] for k,v in candidates.items()},
        'fold_rmse_pixels':[rmse(chosen['oof'][data['folds']==f],target[data['folds']==f]) for f in range(5)],
        'validation':'Five group folds with identical decoded images kept together. For each training image only predictions from its held-out model are used. Small hyperparameter/readout selection uses these OOF results.',
        'train_images':80,'test_images':20,'rows':40,'columns':['ImageID','value'],
        'gpu':fold_reports[0]['gpu'],'fold_models':fold_reports,
        'final_inference':json.loads((HERE/'inference_check.json').read_text(encoding='utf-8')) if (HERE/'inference_check.json').exists() else None,
        'checkpoint_selection_note':'Fold training logs use AMP evaluation; final OOF/test predictions are recomputed with the inference precision recorded above.',
        'training_seconds_sum':sum(v['training_seconds'] for v in fold_reports),
        'invisible_training_examples':0,
        'visibility_note':'All 80 training labels are visible. No calibrated invisible detector can be validated from these data; this submission uses the localization models without an unvalidated zero-coordinate threshold.',
        'test_accuracy_note':'Test labels are unavailable and Kaggle submission has not been made. OOF RMSE is a local validation result, not an official leaderboard score.',
        'csv_checks':{'all_template_ids_match':True,'template_order_preserved':True,'all_values_finite':True,'all_coordinates_within_original_image':True,'no_extra_index':True,'clipped_image_count':clipped_rows},
        'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'pretrained_weights_sha256':hashlib.sha256((HERE/'resnet18-f37072fd.pth').read_bytes()).hexdigest(),
        'source_archive_sha256':json.loads((HERE/'audit.json').read_text(encoding='utf-8'))['archive_sha256']}
    (OUT/'训练与提交检查.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'折外验证预测.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['data','fold','true_x','true_y','pred_x','pred_y','distance_pixels'])
        for i,p in enumerate(chosen['oof']):
            writer.writerow([int(data['ids'][i]),int(data['folds'][i]),*target[i].tolist(),*p.tolist(),float(np.linalg.norm(p-target[i]))])
    np.save(HERE/'final_test_coordinates.npy',test_pred)
    np.save(HERE/'final_oof_coordinates.npy',chosen['oof'])
    draw_validation(data,chosen['oof'])
    print('FINAL',json.dumps({k:v for k,v in report.items() if k not in ['fold_models']},ensure_ascii=False),flush=True)
    print('CSV_VERIFIED',str(path),flush=True)


def draw_validation(data,pred):
    # These are labeled training images with held-out predictions, not test labels.
    indices=np.argsort(np.linalg.norm(pred-data['target'],axis=1))[::-1][:12]
    canvas=Image.new('RGB',(4*270,3*290),'white')
    draw=ImageDraw.Draw(canvas)
    for k,index in enumerate(indices):
        x0,y0=(k%4)*270,(k//4)*290
        im=Image.fromarray(data['images'][index]).resize((256,256))
        canvas.paste(im,(x0,y0))
        for xy,color in [(data['target'][index],'lime'),(pred[index],'red')]:
            p=(xy-data['crops'][index,:2]+.5)*data['scales'][index]-.5+data['pads'][index]
            px,py=x0+p[0]/2,y0+p[1]/2
            draw.line((px-5,py,px+5,py),fill=color,width=2)
            draw.line((px,py-5,px,py+5),fill=color,width=2)
        distance=np.linalg.norm(pred[index]-data['target'][index])
        draw.text((x0+3,y0+260),f'ID {int(data["ids"][index])}  distance {distance:.1f} px',fill='black')
        draw.text((x0+3,y0+274),'green=label red=OOF prediction',fill='black')
    canvas.save(OUT/'折外预测误差检查.png')


if __name__=='__main__':
    main()
