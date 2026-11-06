# 任务一 植物幼苗分类

模型为ImageNet预训练EfficientNet-B0，采用分层五折和水平翻转概率平均。

- `kaggle_plant_classifier.py`：模型、训练、验证、集成与提交生成。
- `2026task1_kaggle_training.ipynb`：Kaggle执行入口，归档本身没有运行输出。
- `validation_scores.csv`：已保存的五折最优micro-F1，均值0.8960。
- `class_to_idx.json`：五类映射。

从仓库根目录运行 `python task1/kaggle_plant_classifier.py`。通过环境变量PLANT_INPUT_ROOT指定数据根目录，PLANT_WORK_ROOT指定输出目录（本地可设为task1/outputs）。默认五折、每折最多10轮、输入256、批量32。默认数据路径适用于Kaggle；本地运行前必须设置这两个路径。

本任务没有归档逐轮日志或训练权重，不能据默认参数推断历史实际训练轮数。完整环境及方法说明见仓库根README。
