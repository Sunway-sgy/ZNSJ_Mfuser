# MFuser 服务器复现步骤

本文档用于在 Linux 服务器上复现运行 `MFuser` 项目。

## 1. 准备代码

```bash
git clone <your-repo-url>
cd MFuser
```

如果你是从已有目录上传到服务器，确保目录中包含以下关键文件和文件夹：

- `train.py`
- `test.py`
- `configs/`
- `mmseg/`
- `models/`
- `requirements.txt`

## 2. 创建环境

项目 README 推荐使用 `Python 3.9 + PyTorch 2.0.1 + CUDA 11.8`。

```bash
conda create -n mfuser python=3.9 numpy=1.26.4 -y
conda activate mfuser
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
pip install xformers==0.0.20
pip install mmcv-full==1.5.1
pip install mamba_ssm==2.2.2
pip install causal_conv1d==1.4.0
```

如果服务器不能联网，建议先在本地下载好对应的 whl 包或使用镜像源。

## 3. 下载预训练权重

在项目根目录下新建 `pretrained/`：

```bash
mkdir -p pretrained
```

然后将以下权重放进去，文件名要与配置中保持一致：

- `pretrained/dinov2_vitl14_pretrain.pth`
- `pretrained/ViT-L-14-336px.pt`
- `pretrained/EVA02_CLIP_L_336_psz14_s6B.pt`
- `pretrained/siglip_vitl16_384.pth`

不同配置会读取不同的 VLM 权重。例如：

- CLIP 配置使用 `pretrained/ViT-L-14-336px.pt`
- EVA 配置使用 `pretrained/EVA02_CLIP_L_336_psz14_s6B.pt`
- SIGLIP 配置使用 `pretrained/siglip_vitl16_384.pth`

## 4. 准备数据集

README 指向了 TLDR 官方仓库的数据准备方式，建议按其说明先准备好：

- GTA5
- Cityscapes
- BDD100K
- Mapillary

项目默认希望数据目录类似下面这样：

```text
MFuser/
├─ pretrained/
├─ data/
│  ├─ gta/
│  ├─ cityscapes/
│  ├─ bdd100k/
│  └─ mapillary/
```

至少要保证你当前实验所需的源域和目标域数据集存在。

例如 `gta2city` 任务默认使用：

- 源域：`data/gta`
- 目标域：`data/cityscapes`

## 5. 检查并修改数据路径

数据根目录写在 `configs/_base_/datasets/*.py` 中。

例如：

- [`configs/_base_/datasets/gta2city-512.py`](F:\Desktop\experiment_code\MFuser\configs\_base_\datasets\gta2city-512.py)

其中默认值是：

```python
src_dataset_dict = dict(
    type='GTADataset',
    data_root='data/gta',
    ...
)

tgt_dataset_dict = dict(
    type='CityscapesDataset',
    data_root='data/cityscapes',
    ...
)
```

如果你的服务器数据不在项目目录下，需要把 `data_root` 改成绝对路径，例如：

```python
data_root='/data/datasets/gta'
data_root='/data/datasets/cityscapes'
```

## 6. 选择配置文件

配置文件位于：

- `configs/mfuser/`

常见配置示例：

- `configs/mfuser/mfuser_clip_vit-l_1e-4_20k-g2c-512.py`
- `configs/mfuser/mfuser_eva_vit-l_1e-4_20k-g2c-512.py`
- `configs/mfuser/mfuser_siglip_vit-l_1e-4_20k-g2c-512.py`

命名规则大致如下：

- `g2c`: GTA5 -> Cityscapes
- `g2b`: GTA5 -> BDD100K
- `g2m`: GTA5 -> Mapillary
- `c2b`: Cityscapes -> BDD100K
- `c2m`: Cityscapes -> Mapillary

## 7. 单机单卡训练

先用单卡确认环境没问题，推荐执行：

```bash
python train.py configs/mfuser/mfuser_clip_vit-l_1e-4_20k-g2c-512.py
```

训练输出默认保存在：

```text
work_dirs_d/<配置文件名>/
```

例如：

```text
work_dirs_d/mfuser_clip_vit-l_1e-4_20k-g2c-512/
```

## 8. 单机多卡训练

仓库自带的 `dist_train.sh` 里把 `CUDA_VISIBLE_DEVICES=4,5` 写死了，直接使用前建议先改掉，或者直接手动启动。

更稳妥的启动方式：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port=29515 \
  train.py configs/mfuser/mfuser_clip_vit-l_1e-4_20k-g2c-512.py \
  --launcher pytorch --gpus 2 --seed 2023
```

如果你想用 4 张卡，把 `CUDA_VISIBLE_DEVICES` 和 `--nproc_per_node`、`--gpus` 同步改掉即可。

## 9. 评估模型

评估命令示例：

```bash
python test.py \
  configs/mfuser/mfuser_clip_vit-l_1e-4_20k-g2c-512.py \
  work_dirs_d/mfuser_clip_vit-l_1e-4_20k-g2c-512/latest.pth \
  --eval mIoU
```

如果要多卡评估，也不要直接原样使用仓库里的 `dist_test.sh`，因为里面把 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` 写死了。

推荐手动执行：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port=29516 \
  test.py configs/mfuser/mfuser_clip_vit-l_1e-4_20k-g2c-512.py \
  work_dirs_d/mfuser_clip_vit-l_1e-4_20k-g2c-512/latest.pth \
  --launcher pytorch --eval mIoU
```

## 10. 复现时最容易出错的地方

1. `pretrained/` 下权重文件名不一致，导致加载失败。
2. `configs/_base_/datasets/*.py` 中的 `data_root` 没改成服务器实际路径。
3. 直接运行 `dist_train.sh` 或 `dist_test.sh`，但服务器 GPU 编号与脚本中写死的不一致。
4. `mmcv-full==1.5.1`、`mamba_ssm==2.2.2`、`causal_conv1d==1.4.0` 安装失败。
5. 训练时显存不足，需要减小配置中的 `samples_per_gpu`。

## 11. 建议的最小复现流程

如果你只是想先确认代码能跑通，建议按下面顺序做：

1. 建好 `conda` 环境并装完依赖。
2. 下载一种配置需要的预训练权重，例如 CLIP 版本。
3. 准备 `GTA5 + Cityscapes` 数据。
4. 修改 [`configs/_base_/datasets/gta2city-512.py`](F:\Desktop\experiment_code\MFuser\configs\_base_\datasets\gta2city-512.py) 中的 `data_root`。
5. 运行单卡训练命令。
6. 成功后再切换到多卡训练。
7. 训练结束后用 `test.py` 做评估。

## 12. 推荐首条验证命令

如果你要先验证最常见的 `GTA5 -> Cityscapes + CLIP` 配置，可以直接跑：

```bash
python train.py configs/mfuser/mfuser_clip_vit-l_1e-4_20k-g2c-512.py
```

如果这条命令报错，优先检查：

- Python / CUDA / PyTorch 版本是否匹配
- 权重文件是否存在
- 数据路径是否正确
- 显卡数量和显存是否足够
