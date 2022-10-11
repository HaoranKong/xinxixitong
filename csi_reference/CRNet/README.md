## Overview

这是论文 [Multi-resolution CSI Feedback with deep learning in Massive MIMO System](https://arxiv.org/abs/1910.14322) 的Pytorch代码实现。

## Requirements

为运行该代码，需要具备以下工作条件：

- Python >= 3.7
- [PyTorch >= 1.2](https://pytorch.org/get-started/locally/)
- [thop](https://github.com/Lyken17/pytorch-OpCounter)

建议使用anaconda虚拟环境，在linux系统下进行环境配置。

建议使用支持CUDA的GPU进行模型训练，以利用显卡的并行计算能力加快模型训练。也可以使用CPU+内存进行模型训练。

为适应设备内存，可以通过降低config中的batch_size值（即批处理样本数量）。作为参考，batch_size为200时，使用的内存大小约为25GB（CPU计算模式）。

## Project Preparation

#### A. Data Preparation

本课程提供的CSI数据由 [open source library of COST2100](https://github.com/cost2100/cost2100) 生成，为无线信道的小尺度衰落角度时延域（裁切）的CSI数据

数据集可以从百度网盘中获取，链接为https://pan.baidu.com/s/1VfEUzPN4Xy8jDulMH5Prog，提取码为a6gs

请将数据集按照以下目录结构放置：

```
-- csi_reference
 |-- CRNet
   |-- checkpoints          <= 将预训练模型置于此
   |-- dataset
     |-- csi_datasets.py
     |-- csiDatasets        <= 将数据集置于此
       |-- indoor               <= 室内数据集
       |-- outdoor              <= 室外数据集
     |-- models             <= 模型结构
     |-- utils
     |-- main.py            <= 主函数
 |-- modelSave_stack.py     <= 模型保存工具
```

数据维度为dataset_length * C * Nt * Nc'，其中：
    
* dataset_length: 数据集中的总样本数，其中数据集样本数为100000，验证集样本数为20000
* C：角度时延域CSI矩阵的通道数，C=2，即实部+虚部
* Nt：角度域的采样点数量，数值等同于BS端的天线数，本数据集中Nt=32
* Nc': 时延域的采样点数量，数值远小于OFDM的子载波数量，一般设定与Nt值相等，本数据集中Nc'=32

每个批处理过程中，批处理数据维度为batch_size * C * Nt * Nc'，其中batch_size为批处理样本数量

在送入神经网络模型前，需要进行数据预处理。数据预处理方法不固定，目标为将CSI数据集中的数据大小放缩到[0, 1]之间。
请在/CRNet/dataset/csi_datasets.py的预留空白板块中进行预处理方法的设计。


## CRNet的模型训练

1. 在main.py函数中修改工作目录、数据集所在目录、训练参数等
2. 在csi_datasets.py函数中设计归一化方法
3. 利用pycharm的debug/run进行调试/运行，或在linux终端使用python main.py进行代码运行。为防止出现训练短线和数据丢失，建议后台运行程序。

在训练过程中，每训练一个epoch就会进行一次验证集测试，同时输出当前epoch的NMSE结果。如果需要修改测试频次，请在/CRNet/utils/solver.py中把第21行”val_freq“与”test_freq"的数值修改为所需要的频次


## 模型修改与最终性能测试

1. 如需修改模型结构或模型参数，请在/CRNet/models/crnet.py中进行修改，或自行新建一个工作目录（结构与CRNet工作目录相同），并在models子目录下的对应文件设计自己的网络结构
2. 本数据集仅在期末验收时由老师提供最终性能测试的测试集，因此请在代码中提前预留测试集的测试代码，以便验收时使用。要求可以利用测试集获得NMSE结果。
   * 测试集也是利用cost2100信道模型生成，同样是未经过归一化的角度时延域（裁切）数据，样本数量为30000。
   * 若最终测试集的性能与验证集的性能相差过远，则存在作弊风险，影响最终成绩，因此请认真进行方案设计