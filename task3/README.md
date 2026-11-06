# 任务三 眼底中心凹定位

模型为ImageNet ResNet18编码器与热图解码器，使用局部期望坐标读出，最终五折平均。

- `models.py`：网络、损失、增强与坐标变换。
- `prepare.py`：图像处理和固定分组划分。
- `run.py`：冻结特征岭回归基线及热图模型训练。
- `infer_saved.py`、`finalize.py`：确定性FP32推理与提交检查。
- `heatmap_s2026_fold0`至`fold4`：已归档的逐轮历史和结果。
- `outputs/`：归档检查报告及新生成的40行坐标提交。

初始化权重路径为task3/resnet18-f37072fd.pth，官方下载地址及完整哈希见根README。从仓库根目录运行：

```text
python task3/prepare.py --archive "2026task3.zip"
python task3/run.py --stage baseline --batch-size 8
python task3/run.py --stage train --folds 0,1,2,3,4 --epochs 180 --batch-size 8
python task3/infer_saved.py --accept-new-precision
python task3/infer_saved.py
python task3/finalize.py
```

重新训练前先备份并移走已归档的五折结果目录，否则训练程序会跳过已有result.json的折。全部运行数据位于task3内，输出位于task3/outputs。16.20465691像素是本地OOF坐标RMSE，不是线上得分。原始比赛图片和大权重不在本仓库。
