# 毕业论文-基于物理信息启发的双分支Swin Transformer野火风险预测研究

本毕业论文基于LOAN(https://arxiv.org/abs/2212.08208, https://github.com/HakamShams/LOAN) 的代码及江丰楗学长的工作构建。

## Setup
配置虚拟环境。
对于conda，可以使用yml文件安装依赖项：
```
  conda env create -f environment.yml
```

## Code

数据加载。
FireCube数据集的加载方式:
```
  FireCube_dataloader.py
```
对于引入vpd重构的dataloader，位于FireCube_dataloader_vpd.py，因此训练时需导入此dataloader_vpd，可重新配置args进行更简洁的操作。

对于训练：
```
  python train.py
```
对于测试:
```
  python test.py
```

参考args配置选择相应的模型。
- models/DualBranchSwin2d3d.py 使用Swin Transformer 2D处理静态分支，使用Swin Transformer 3D处理动态分支，Geo-CA进行融合。
- models/DualBranchSwinGCA_uv.py 在上述基础上引入风场矢量，同样可通过配置参数进行简化。
- models/DualBranchSwinGCA.py 原始采用CNN处理静态分支。
- models/DualBranchSwinAdaLN.py 原始采用CNN处理静态分支，使用AdaLN对动态和静态变量进行融合。

在进行训练和测试时需注意各种配置。

## Dataset

要在 FireCube 数据集上进行训练，可以从以下网址下载训练/测试样本 https://zenodo.org/record/6528394 (~250GB).

压缩 datasets.tar.gz 并复制文件 [mean_std_train.json](data/mean_std_train.json) 到目录 datasets/datasets_grl/npy/spatiotemporal。如果是同时使用时间和空间，即静态和动态特征，理论上只需要spatiotemporal这一个文件夹的数据。

如果要在另一个数据集上进行训练，需要创建一个像 [FireCube_dataloader.py](FireCube_dataloader.py) 的新的数据加载器文件。

## Train
训练使用swanlab可视化，位于 https://swanlab.cn/@Micro/Graduation_Paper?utm_source=website_qr&utm_medium=qr_scan ,可查看每次的训练参数及Loss、Precision、Recall、F1-Score、AUROC、OA等曲线。

## Checkpoints
位于 log/test_SwinGCA_{}/model_checkpoints 下，也可参考LOAN文章训练出的pretrained models。

## Paper
毕业论文的latex代码位于Paper/目录下.

## Acknowledgements
感谢。
