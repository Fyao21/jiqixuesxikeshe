# 四项图像学习课程实验

本目录保存四个任务的原始实现、超参数及已留存的实验记录。只上传代码和指标，不包含原始比赛图像、数据包或测试标签。报告中的数字来自这些记录。

| 任务 | 方法 | 本地结果 | 指标口径 |
| --- | --- | --- | --- |
| 1 植物幼苗分类 | ImageNet EfficientNet-B0，分层五折 | 0.8960 micro-F1 | 五折最优分数平均，逐折记录 0.91/0.89/0.89/0.91/0.88 |
| 2 人脸表情分类 | 从头训练小图 ResNet18，EMA | 0.67778366 accuracy | 分组五折中仅第一折，水平翻转平均；全量模型无独立验证分数 |
| 3 中心凹定位 | ResNet18 U-Net 热图，局部坐标读出 | 16.20465691 像素 RMSE | 五折 OOF，原图 X/Y 坐标合并计算 |
| 4 血管分割 | ResNet18 U-Net，全分辨率跳连 | 0.82911402 Dice | 五折 OOF，8视角，阈值0.45；验证参与选型 |

这些不是 Kaggle 线上得分。任务1输出仅保留分数汇总与提交CSV，没有逐轮日志和训练权重，无法追溯其实际停止轮次；Notebook 是无输出的执行入口。其他任务的逐轮历史位于 work/。任务3、4的轮次选择使用混合精度，最终分数已重算为确定性FP32。

## 环境

任务2—4已在 Windows、RTX4060 Laptop 8GB、Python3.12.8、PyTorch2.7.1+cu126、torchvision0.22.1+cu126 上运行。任务1实现支持 Kaggle GPU，历史执行环境未留在所提供目录中。任务1额外需要pandas；任务2—4具体环境见对应requirements和结果文件。

## 运行顺序

### 任务1

把比赛数据放入含 train/类别/图像、test/图像、示例提交CSV的目录。设置 PLANT_INPUT_ROOT、PLANT_WORK_ROOT（可选 PLANT_PRETRAINED_PATH），运行 `python task1/kaggle_plant_classifier.py`。默认五折、至多10轮、256像素、批量32；各折最优状态仅保存在内存中，原实现没有保存训练权重。脚本完成后输出 submission.csv、class_to_idx.json 和 validation_scores.csv。

### 任务2

`work/prepare_data.py` 的 SOURCE 常量指向原机器 `C:\Users\w2922\Downloads\2026task2.zip`；其他机器请仅将此数据路径改为原数据包位置。依次运行：

```text
python work/prepare_data.py
python work/train_gpu.py --mode validate --epochs 70 --batch-size 256
python work/train_gpu.py --mode full --epochs 40 --schedule-epochs 70 --batch-size 256
python work/verify_submission.py
```

注意不要把全量模型在训练过的数据上的分数作为验证分数。分组数据审计及两次训练历史均原样保留。

### 任务3

从官方地址 `https://download.pytorch.org/models/resnet18-f37072fd.pth` 下载 ImageNet 初始化权重到 `work/task3/resnet18-f37072fd.pth`，完整SHA256为 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。

```text
python work/task3/prepare.py --archive "2026task3.zip"
python work/task3/run.py --stage baseline --batch-size 8
python work/task3/run.py --stage train --folds 0,1,2,3,4 --epochs 180 --batch-size 8
python work/task3/infer_saved.py --accept-new-precision
python work/task3/infer_saved.py
python work/task3/finalize.py
```

原训练程序遇到已有 result.json 会跳过该折。重新训练前请备份并移走 `work/task3/heatmap_s2026_fold0` 至 `fold4`，再开始运行。原始脚本保留用于复查，未为报告修改算法。若使用已交付的本地模型复现包，则无需重新训练。

### 任务4

将相同ResNet18初始化权重放到 `work/task4/resnet18-f37072fd.pth`。先备份并移走已留存的 `work/task4/fold0` 至 `fold4`（训练脚本会跳过有 result.json 的目录）。

```text
python work/task4/prepare.py --archive "2026task-4.zip"
python work/task4/train.py --folds 0,1,2,3,4 --epochs 220 --batch-size 4
python work/task4/infer.py
python work/task4/infer.py
python work/task4/finalize.py
```

第二次推理用于比较同机、同环境的确定性结果。保留的 prediction_hashes.json 为旧模型参考哈希，重新训练模型不应拿旧模型哈希强行比较；移走旧fold目录后重新建立参考。任务4按原数据包转换函数，将标签双线性缩放为512×512后把非零像素编码为前景，血管为白色。网页描述的极性与实际标签相反，报告已说明。

## 审计材料

`source_manifest.json` 记录原代码与记录文件的SHA256。文件夹中的训练结果与历史是已完成实验的记录；代码为既有模型方法的课程实现，不声称提出EfficientNet、ResNet或U-Net。任务3/4大权重在已交付的模型复现ZIP中，没有将大权重或原始比赛数据上传本仓库。
