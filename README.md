# 基于改进 YOLOv8n 的 VisDrone 航拍图像目标检测研究

**English Title:** A Study of Aerial Image Object Detection Based on Improved YOLOv8n

## 1. 项目简介

本项目以 YOLOv8n 为基线模型，基于 VisDrone 航拍图像目标检测数据集，研究 EMA 注意力机制和改进 SPPF-S 特征提取模块对复杂环境下目标检测性能的影响。

航拍图像中的目标通常具有尺寸小、分布密集、背景复杂、遮挡严重等特点，给目标检测算法带来了较大挑战。本项目通过模型结构调整和对比实验，分析不同改进方案对检测精度和推理速度的影响。

本项目包含四组实验：

1. YOLOv8n Baseline 基线模型
2. YOLOv8n + EMA 注意力机制
3. YOLOv8n + EMA + SPPF-S
4. YOLOv8n + EMA + SPPF-S（ReLU）

**实验说明：** 当前实验结果表明，三组改进模型均未超过 Baseline 的检测指标。因此，本项目重点分析改进模块在当前数据集、训练配置和实验条件下的实际表现，不将未取得的性能提升表述为实验结论。

## 2. 实验环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows |
| Python | 3.11.15 |
| PyTorch | 2.5.0 |
| CUDA | 11.8 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU 6GB |
| Ultralytics | 8.4.115 |
| 检测模型 | YOLOv8n |
| 数据集 | VisDrone |
| 输入尺寸 | 640 × 640 |

## 3. 数据集

本项目使用 VisDrone 航拍图像目标检测数据集，涉及以下十类目标：

- pedestrian
- people
- bicycle
- car
- van
- truck
- tricycle
- awning-tricycle
- bus
- motor

数据集采用 YOLO 格式组织。项目提供数据转换和数据检查脚本，但不直接上传原始数据集。

数据集目录示例：

```text
datasets/
└── VisDrone/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

数据集配置文件：`configs/visdrone.yaml`

数据转换脚本：`scripts/convert_visdrone.py`

数据检查脚本：`scripts/check_dataset.py`

请自行从 VisDrone 官方渠道获取数据集，并按照实际数据路径修改 YAML 配置文件。

## 4. 模型改进方法

### 4.1 EMA 注意力机制

EMA（Efficient Multi-Scale Attention）是一种轻量化注意力机制，通过跨空间信息交互和多尺度特征建模，增强网络对重要目标区域的特征表达能力。

本项目将 EMA 引入 YOLOv8n 网络，研究其对航拍图像中小目标检测效果的影响。

### 4.2 SPPF-S 特征提取模块

SPPF-S 是本项目实验使用的改进特征提取模块，旨在调整多尺度特征提取方式，增强网络对不同尺度目标的特征表达能力。

本项目将其与 EMA 组合，开展联合实验。

### 4.3 ReLU 激活函数实验

在 EMA + SPPF-S 实验基础上，进一步开展 ReLU 激活函数相关实验，用于分析激活函数设置对模型检测性能的影响。

## 5. 实验结果

四组模型在验证集上的指标如下：

| 模型 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Baseline | 46.64% | 34.34% | 32.38% | 18.40% |
| EMA | 45.15% | 33.66% | 31.70% | 17.32% |
| EMA + SPPF-S | 44.17% | 32.67% | 30.78% | 17.06% |
| EMA + SPPF-S（ReLU） | 43.10% | 32.01% | 29.49% | 16.22% |

### 5.1 实验分析

从当前实验结果可以看出：

1. Baseline 模型取得了四组实验中最高的 Precision、Recall、mAP50 和 mAP50-95。
2. EMA 单独引入后，检测指标有所下降，说明注意力机制的加入并不必然带来性能提升。
3. EMA 与 SPPF-S 组合后，当前实验结果进一步下降，可能与特征融合方式、网络结构适配和训练参数有关。
4. ReLU 实验的指标低于前三组模型，但由于该实验与初始 SPPF-S 实验的训练轮次等条件并不完全一致，不能将差异简单归因于激活函数本身。

因此，当前实验说明改进模块的有效性需要结合网络结构、训练策略和数据集特点进行综合分析。

### 5.2 实验结果文件

各组实验的训练曲线、混淆矩阵、PR 曲线和 CSV 指标文件存放在 `results/` 目录下。

## 6. 项目目录结构

```text
YOLOv8_VisDrone/
│
├── configs/
│   ├── visdrone.yaml
│   ├── yolov8n_ema.yaml
│   └── yolov8n_ema_sppfs.yaml
│
├── scripts/
│   ├── check_dataset.py
│   ├── convert_visdrone.py
│   ├── train_baseline.py
│   ├── train_ema.py
│   ├── train_ema_sppfs.py
│   └── train_ema_sppfs_relu.py
│
├── models/
│   └── ultralytics_custom/
│       ├── block.py
│       ├── modules_init.py
│       └── tasks.py
│
├── weights/
│   ├── baseline_best.pt
│   ├── ema_best.pt
│   ├── ema_sppfs_best.pt
│   └── ema_sppfs_relu_100_best.pt
│
├── results/
│   ├── baseline/
│   ├── ema/
│   ├── ema_sppfs/
│   ├── ema_sppfs_relu_100/
│   └── metrics.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

注：`datasets/`、`runs/` 和 `backup/` 等目录不上传至 GitHub。模型源码备份文件用于记录自定义模块及解析器修改，不是可直接独立运行的 Ultralytics 安装包。

## 7. 安装与运行

### 7.1 创建环境

建议使用 Python 3.11。

```bash
conda create -n yolo python=3.11 -y
conda activate yolo
```

### 7.2 安装 PyTorch

本项目实验环境使用 PyTorch 2.5.0 和 CUDA 11.8。可使用以下命令安装对应版本：

```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
```

### 7.3 安装其他依赖

```bash
pip install ultralytics==8.4.115
pip install numpy opencv-python PyYAML matplotlib pandas Pillow
```

### 7.4 数据准备

下载 VisDrone 数据集，完成格式转换，并检查标签和图像路径。

```bash
python scripts/convert_visdrone.py
python scripts/check_dataset.py
```

请根据脚本实际参数和本地数据集路径进行配置。

### 7.5 模型训练

在项目根目录执行：

```bash
python scripts/train_baseline.py
```

```bash
python scripts/train_ema.py
```

```bash
python scripts/train_ema_sppfs.py
```

```bash
python scripts/train_ema_sppfs_relu.py
```

注意：自定义 EMA 和 SPPF-S 模块依赖相应的 Ultralytics 源码修改。复现实验前，需要按照本项目保存的源码备份，将自定义模块定义、模块导出和模型解析器修改应用到匹配版本的 Ultralytics 安装目录中。

以上脚本记录实验训练配置；完整复现实验还需要使用相同的数据集划分、预训练权重、源码修改和运行参数。

## 8. 模型权重

本项目提供四组实验的最佳模型权重：

| 权重文件 | 对应模型 |
|---|---|
| `baseline_best.pt` | YOLOv8n Baseline |
| `ema_best.pt` | YOLOv8n + EMA |
| `ema_sppfs_best.pt` | YOLOv8n + EMA + SPPF-S |
| `ema_sppfs_relu_100_best.pt` | YOLOv8n + EMA + SPPF-S（ReLU） |

### 加载模型示例

```python
from ultralytics import YOLO

model = YOLO("weights/baseline_best.pt")

results = model.predict(
    source="test.jpg",
    imgsz=640,
    conf=0.25,
    save=True
)
```

自定义结构权重需要对应的自定义模块代码和解析器环境，不能保证在未经修改的原版 Ultralytics 环境中直接加载。

## 9. 实验结果复核

可以使用以下方式对模型进行验证：

```python
from ultralytics import YOLO

model = YOLO("weights/baseline_best.pt")

metrics = model.val(
    data="configs/visdrone.yaml",
    imgsz=640,
    batch=4,
    device=0
)

print(metrics)
```

其他模型只需替换权重路径，并确保对应的自定义模块已正确加载。

## 10. 研究结论与不足

本项目完成了 YOLOv8n 基线模型及三种改进方案的训练、验证和对比分析。

实验表明，在当前实验条件下，EMA、SPPF-S 及其组合尚未实现对基线模型的性能超越。这说明轻量化注意力机制和特征提取模块的设计需要充分考虑网络结构、参数量、特征融合和训练策略之间的关系。

后续可从以下方面进一步研究：

- 优化注意力模块的插入位置；
- 分析不同尺度特征融合策略；
- 调整训练轮次、学习率和优化器；
- 进一步开展模块消融实验；
- 在统一训练条件下进行更多重复实验；
- 结合小目标检测特点研究更有效的特征增强方法。


## 11. 参考文献

1. Ultralytics. YOLOv8 Documentation. https://docs.ultralytics.com/
2. VisDrone Dataset. https://github.com/VisDrone/VisDrone-Dataset
3. Ouyang D, et al. Efficient Multi-Scale Attention Module with Cross-Spatial Learning. ICASSP, 2023.
4. Hu J, Shen L, Sun G. Squeeze-and-Excitation Networks. CVPR, 2018.

---

**项目主题：** 基于改进 YOLOv8n 的 VisDrone 航拍图像目标检测研究

**主要内容：** YOLOv8n、EMA 注意力机制、SPPF-S 特征提取模块、航拍小目标检测、消融实验与性能分析。

