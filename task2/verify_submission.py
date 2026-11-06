"""Independently verify final CSV against the original competition archive."""
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
CSV = ROOT / 'outputs' / 'submission.csv'
ARCHIVE = Path(r'C:\Users\w2922\Downloads\2026task2.zip')


def main():
    with zipfile.ZipFile(ARCHIVE) as z:
        template = list(csv.DictReader(io.StringIO(z.read('submission.csv').decode('utf-8-sig'), newline='')))
        test_ids = {Path(n).name for n in z.namelist() if '/test/' in n and n.lower().endswith('.jpg')}
    with CSV.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ['ID', 'Emotion'], reader.fieldnames
        rows = list(reader)
    ids = [r['ID'] for r in rows]
    assert len(rows) == len(template) == len(test_ids) == 7065
    assert len(set(ids)) == len(ids)
    assert ids == [r['ID'] for r in template], 'Template row order mismatch'
    assert set(ids) == test_ids, 'Test ID set mismatch'
    assert all(r['Emotion'] in {'0', '1', '2', '3', '4', '5'} for r in rows)
    assert all(len(r) == 2 for r in rows), 'Extra CSV columns'
    with CSV.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    validate = json.loads((ROOT/'result_validate.json').read_text(encoding='utf-8'))
    full = json.loads((ROOT/'result_full.json').read_text(encoding='utf-8'))
    audit = json.loads((ROOT/'data_audit.json').read_text(encoding='utf-8'))
    report = {
        'file': CSV.name, 'rows': len(rows), 'columns': ['ID', 'Emotion'], 'sha256': digest,
        'all_template_and_test_ids_matched': True, 'template_order_preserved': True,
        'no_duplicate_or_missing_ids': True, 'all_labels_in_0_to_5': True, 'no_extra_index_column': True,
        'prediction_counts': dict(sorted(Counter(r['Emotion'] for r in rows).items())),
        'gpu': full['config']['gpu'], 'torch': full['config']['torch'],
        'cuda_runtime': full['config']['cuda_runtime'],
        'model': full['config']['architecture'],
        'label_order': full['config']['label_order'],
        'validation_training_samples': validate['config']['fit_count'],
        'validation_samples': validate['config']['validation_count'],
        'best_validation_epoch': validate['best']['epoch'],
        'validation_accuracy_without_tta': validate['validation_accuracy'],
        'validation_accuracy_with_horizontal_flip_tta': validate['validation_flip_tta_accuracy'],
        'validation_confusion_matrix': validate['confusion_matrix'],
        'validation_grouping': audit['validation_split'],
        'final_training_samples': full['config']['fit_count'],
        'final_training_epochs': full['epochs_completed'],
        'final_schedule_horizon': full['config']['schedule_epochs'],
        'final_horizontal_flip_tta': full['use_flip_tta'],
        'total_training_seconds': validate['training_seconds']+full['training_seconds'],
        'source_archive_sha256': audit['source_archive_sha256'],
        'score_note': 'Validation scores come from the grouped holdout selection run. The final model used all training labels. Kaggle test accuracy has not been measured because this CSV has not been submitted.'
    }
    (ROOT/'outputs/训练与CSV检查结果.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
