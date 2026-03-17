# SPAG — 图像 Embedding 抗反演隐私保护系统

> **S**parse **P**rivacy-region selection + **A**dversarial perturbation for image embeddin**G**s

## 1. 项目概述

在"客户端本地编码图像 → 上传 embedding → 云端向量数据库检索"的场景下，embedding 可能被攻击者反演重建出原始图像，造成隐私泄露。

本项目在**客户端侧**部署前置防御：通过 **遮蔽打分选择隐私关键区域** + **重建引导的局部对抗扰动（Masked PGD）** 生成受保护 embedding，在保留检索效用的同时显著降低反演重建质量。

### 核心流程

```text
原始图像 x
  │
  ├─ ① CLIP 编码器 E  ──→  z = E(x)          原始 embedding
  │
  ├─ ② 遮蔽打分 (OcclusionSelector)
  │     对每个 patch 遮蔽后计算：
  │       u_j  = 语义代价 (embedding 漂移)
  │       p_j  = 隐私收益 (重建损失增量)
  │       s_j  = p_j / (u_j + ε)    ← 综合得分
  │     选 top-k patch → 二值 mask M
  │
  ├─ ③ Masked PGD 扰动 (MaskedPGD)
  │     仅对 M 标记区域施加对抗扰动 δ
  │     优化: max L_rec(R(E(x')), x) - λ_u·drift - λ_s·TV
  │     得到 x' = clip(x + M ⊙ δ, 0, 1)
  │
  └─ ④ 受保护 embedding  z' = E(x')          上传到云端
```

---

## 2. 项目结构

```text
GraduationProject/
├── configs/
│   └── default.yaml              # 所有超参数统一配置
├── spag/                          # 核心库
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py            # 惰性导入
│   │   ├── encoder.py             # CLIPEncoder — CLIP ViT-B/16 封装
│   │   ├── reconstructor.py       # Reconstructor — 影子反演 CNN 解码器
│   │   ├── selector.py            # OcclusionSelector — 遮蔽打分 + top-k 选择
│   │   └── perturber.py           # MaskedPGD — 局部对抗扰动生成
│   ├── eval/
│   │   ├── __init__.py
│   │   └── metrics.py             # PSNR / SSIM / LPIPS / cosine drift
│   └── data/
│       ├── __init__.py
│       └── dataset.py             # ImageDataset (递归加载图片目录)
├── scripts/
│   ├── train_reconstructor.py     # Step 1: 训练影子反演模型
│   ├── run_defense.py             # Step 2: 运行 SPAG 防御 + 评估
│   └── eval_baseline.py           # Step 3: 对比基线实验
├── reproduce/                     # 参考文献复现代码
│   ├── ID3PM/                     # 基于扩散模型的反演器
│   └── IdDecoder/                 # 基于 StyleGAN 的反演器
├── design.md                      # 详细设计文档
├── requirements.txt               # Python 依赖
└── README.md                      # ← 本文件
```

---

## 3. 各模块说明

### 3.1 CLIPEncoder ([spag/models/encoder.py](spag/models/encoder.py))

- 使用 `open_clip` 加载 `ViT-B/16 (OpenAI)` 预训练权重
- 输出 512 维 L2 归一化 embedding
- 提供 `encode()` (无梯度) 和 `encode_with_grad()` (保留计算图供 PGD 反传)
- 所有参数冻结，不参与训练

### 3.2 Reconstructor ([spag/models/reconstructor.py](spag/models/reconstructor.py))

- **角色**：影子反演模型 (Shadow Reconstructor)，模拟攻击者
- **架构**：FC 投影 → 5 层 ConvTranspose2d + ResBlock（512→256→128→64→32→3）
- **输入**：512 维 embedding → **输出**：224×224 RGB 图像 [0,1]
- **训练目标**：L1 + λ·LPIPS（不使用 GAN 以保证训练稳定）

### 3.3 OcclusionSelector ([spag/models/selector.py](spag/models/selector.py))

- 将图像分为 16×16 patch（对齐 ViT patch 划分）
- 逐个遮蔽 patch 后计算：
  - **u_j**（语义代价）= 1 − cos(E(x_{-j}), E(x))
  - **p_j**（隐私收益）= L1_rec(R(E(x_{-j})), x) − L1_rec(R(E(x)), x)，负值 clamp 为 0
  - **s_j** = p_j / (u_j + ε)
- 选择 s_j 最高的 top-k patch，生成二值 mask
- 支持三种遮蔽模式：mean / zero / blur

### 3.4 MaskedPGD ([spag/models/perturber.py](spag/models/perturber.py))

- 仅在 mask 标记区域内执行 PGD 优化
- 优化目标（最小化）：
  ```
  L = −L_rec(R(E(x')), x) + λ_u·(1 − cos(E(x'), z)) + λ_s·TV(M⊙δ)
  ```
  即：最大化重建误差 + 最小化 embedding 漂移 + 平滑正则
- 扰动约束：L∞ ≤ ε（默认 ~8/255）
- 默认 10 步 PGD

### 3.5 评估指标 ([spag/eval/metrics.py](spag/eval/metrics.py))

| 指标 | 类型 | 防御成功方向 | 说明 |
|------|------|:---:|------|
| PSNR | 隐私 | ↓ | 重建图 vs 原图的峰值信噪比 |
| SSIM | 隐私 | ↓ | 结构相似性 |
| LPIPS | 隐私 | ↑ | 感知距离 |
| MSE | 隐私 | ↑ | 均方误差 |
| cos_sim | 效用 | → 1 | embedding 余弦相似度（越高越好） |
| cos_drift | 效用 | → 0 | embedding 漂移（越低越好） |

---

## 4. 快速上手

### 4.1 环境准备

```bash
# 创建 conda 环境（已有可跳过）
conda create -n spag python=3.10 -y
conda activate spag

# 安装依赖
pip install -r requirements.txt

# Windows 用户如遇到 OpenMP 冲突，设置环境变量：
set KMP_DUPLICATE_LIB_OK=TRUE
```

**核心依赖**：PyTorch ≥ 1.10、open_clip_torch、lpips、pytorch_msssim

### 4.2 Step 1 — 训练影子反演模型

准备一个图片目录（任意格式 jpg/png/...），越贴近你的业务域效果越好。

```bash
python scripts/train_reconstructor.py --data_dir reproduce/IdDecoder/celeba_hq/train --output_dir checkpoints --config configs/default.yaml
```

- 默认训练 50 epoch，每 5 epoch 保存一次，同时实时保存当前最优模型
- 支持 `--resume checkpoints/reconstructor_epoch25.pth` 断点恢复
- 推荐产出：`checkpoints/reconstructor_best.pth`（用于后续防御与基线对比）

### 4.3 Step 2 — 运行 SPAG 防御

```bash
python scripts/run_defense.py --data_dir reproduce/IdDecoder/celeba_hq/val --reconstructor_ckpt checkpoints/reconstructor_best.pth --output_dir results/defense --num_images 100 --num_vis 20
```

- 对每张图自动执行：打分 → 选区 → PGD 扰动 → 评估指标
- 前 20 张保存可视化到 `results/defense/visualizations/`
- 可视化每行：原图 | 扰动图 | mask | 扰动放大 | 原始重建 | 防御后重建
- 汇总指标保存到 `results/defense/metrics.json`

### 4.4 Step 3 — 基线对比实验

```bash
python scripts/eval_baseline.py \
  --data_dir <测试图片目录> \
  --reconstructor_ckpt checkpoints/reconstructor_best.pth \
  --output_dir results/baselines \
  --num_images 50
```

将以下 5 种方法进行横向对比：
1. **no_defense** — 无防御（攻击上界）
2. **gaussian_img** — 图像加高斯噪声
3. **gaussian_emb** — embedding 加高斯噪声
4. **random_mask** — 随机选区 + PGD
5. **spag** — 本方法（打分选区 + PGD）

---

## 5. 配置文件说明

所有超参数集中在 `configs/default.yaml`：

```yaml
encoder:
  model_name: "ViT-B-16"     # CLIP 模型名称
  pretrained: "openai"        # 预训练来源

reconstructor:
  embed_dim: 512              # embedding 维度
  base_channels: 512          # 解码器基础通道数

training:
  batch_size: 32              # 训练 batch size
  learning_rate: 0.0001       # Adam 学习率
  num_epochs: 50              # 训练轮数
  lambda_lpips: 0.5           # LPIPS 损失权重

selector:
  patch_size: 16              # patch 大小（对齐 ViT-B/16）
  top_k_ratio: 0.15           # 选择 15% 的 patch
  occlusion_mode: "mean"      # 遮蔽方式: mean / zero / blur
  eps: 0.001                  # 分母稳定项

perturber:
  epsilon: 0.03               # L∞ 扰动预算 (≈8/255)
  alpha: 0.005                # PGD 单步步长
  num_steps: 10               # PGD 迭代次数
  lambda_util: 1.0            # 效用保持权重
  lambda_smooth: 0.01         # TV 平滑正则权重
```