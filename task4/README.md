# 任务四 眼底血管分割

模型为预训练ResNet18 U-Net，带全分辨率跳连；最终五折平均、八视角、阈值0.45。

- `model.py`：网络、增强、损失和预测。
- `prepare.py`：数据处理和整图五折划分。
- `train.py`：五折混合精度训练与EMA选模。
- `infer.py`、`finalize.py`：确定性FP32推理、RLE编码及独立检查。
- `fold0`至`fold4`：已归档历史、结果和参考概率哈希。
- `outputs/`：归档检查报告及新生成的20行分割提交。

将初始化权重放入task4/resnet18-f37072fd.pth。重新训练前先备份并移走fold0至fold4；否则已有result.json会使程序跳过训练。从仓库根目录运行：

```text
python task4/prepare.py --archive "2026task-4.zip"
python task4/train.py --folds 0,1,2,3,4 --epochs 220 --batch-size 4
python task4/infer.py
python task4/infer.py
python task4/finalize.py
```

第二次推理核对同机数值一致性；重新训练不可直接比较旧权重的参考概率哈希。缓存和权重位于task4，提交位于task4/outputs。

标签按附带官方代码双线性缩放至512×512后取非零为前景，白色为血管，按行展开、从1开始编码。0.82911402是参与选型的本地OOF Dice，不是线上测试得分。
