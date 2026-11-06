# 任务二 人脸表情分类

模型为从头训练的灰度小图ResNet18，使用EMA和水平翻转平均。

- `prepare_data.py`：数据检查及重复图像分组划分。
- `train_gpu.py`：模型、单折验证、全量重训和预测。
- `verify_submission.py`：提交格式与ID检查。
- `history_validate.json`、`history_full.json`、`result_*.json`：已完成运行记录。
- `outputs/`：已归档检查报告及后续生成的提交文件。

在仓库根目录依次运行：

```text
python task2/prepare_data.py
python task2/train_gpu.py --mode validate --epochs 70 --batch-size 256
python task2/train_gpu.py --mode full --epochs 40 --schedule-epochs 70 --batch-size 256
python task2/verify_submission.py
```

原数据路径在prepare_data.py的SOURCE和verify_submission.py的ARCHIVE常量中，其他机器请改为实际ZIP位置。运行缓存、权重、历史位于task2，提交位于task2/outputs。复查历史记录前请备份目录，避免重新运行覆盖原记录。

0.67778366是分组五折中第一折的验证准确率，不是五折平均或线上测试分数。
