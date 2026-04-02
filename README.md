# Model — 图像局部隐私保护系统

本项目用于“客户端本地编码图像 → 上传 embedding → 云端检索”场景下的抗反演防御。  
核心目标：只保护真正敏感、且对重建攻击有贡献的局部区域，避免整图一刀切扰动。

## 1. 当前实现与设计对应关系

1. **影子重建模型训练**：`Model/models/reconstructor.py` + `scripts/train_reconstructor.py`
2. **块级遮蔽敏感性分析（p_j）**：`Model/models/selector.py` (`OcclusionAnalyzer.compute_sensitivity`)
3. **候选区域筛选（top-M）**：`Model/models/selector.py` (`select_candidates`)
4. **VLM语义隐私判断（q_j）**：`Model/models/vlm.py`（默认 `MockVLMJudge`，可替换真实VLM）
5. **分数融合（s_j）**：`Model/models/fusion.py` (`ScoreFusion`)
6. **局部自适应保护**：`Model/models/protector.py` (`AdaptiveProtector`)

融合核心：

$$
s_j = f(p_j, q_j)
$$

默认实现支持乘性融合与线性融合（见 `configs/default.yaml` 的 `fusion.mode`）。

---

## 2. 目录结构（当前）

```text
GraduationProject/
├── Model/
│   ├── __init__.py
│   ├── data/
│   │   └── dataset.py
│   ├── eval/
│   │   └── metrics.py
│   └── models/
│       ├── encoder.py
│       ├── reconstructor.py
│       ├── selector.py
│       ├── vlm.py
│       ├── fusion.py
│       ├── protector.py
│       └── perturber.py
├── scripts/
│   ├── train_reconstructor.py
│   ├── run_defense.py
│   └── eval_baseline.py
├── configs/
│   └── default.yaml
├── checkpoints/
├── requirements.txt
└── README.md
```

---

## 3. 环境准备

建议 Python 3.10。

```bash
# 1) 创建并激活环境（示例）
conda create -n ml_privacy python=3.10 -y
conda activate ml_privacy

# 2) 安装依赖
pip install -r requirements.txt

# 3) Windows 如遇 OpenMP 冲突
set KMP_DUPLICATE_LIB_OK=TRUE
```

---

## 4. 如何开启影子模型训练（最新）

### 4.1 基本训练命令

```bash
python scripts/train_reconstructor.py \
  --data_dir reproduce/IdDecoder/celeba_hq/train \
  --output_dir checkpoints \
  --config configs/default.yaml
```

### 4.2 常用可选参数

- `--model_type {basic,improved}`：覆盖配置中的重建器类型
- `--resume <ckpt_path>`：从断点恢复训练

示例（断点续训）：

```bash
python scripts/train_reconstructor.py \
  --data_dir reproduce/IdDecoder/celeba_hq/train \
  --output_dir checkpoints \
  --config configs/default.yaml \
  --resume checkpoints/reconstructor_epoch25.pth
```

### 4.3 训练产物

- 最优模型：`checkpoints/reconstructor_best.pth`
- 周期保存：`checkpoints/reconstructor_epoch*.pth`

后续防御和基线评估通常使用 `reconstructor_best.pth`。

---

## 5. 最新脚本使用说明

## 5.1 运行完整防御流程（6阶段）

```bash
python scripts/run_defense.py --data_dir reproduce/IdDecoder/celeba_hq/val --reconstructor_ckpt checkpoints/reconstructor_best.pth --output_dir results/defense --config configs/default.yaml --num_images 10 --num_vis 20
```

输出：

- `results/defense/metrics.json`：总体与逐图像指标
- `results/defense/visualizations/`：可视化结果

## 5.2 基线对比评估

```bash
python scripts/eval_baseline.py \
  --data_dir reproduce/IdDecoder/celeba_hq/val \
  --reconstructor_ckpt checkpoints/reconstructor_best.pth \
  --output_dir results/baselines \
  --config configs/default.yaml \
  --num_images 50 \
  --noise_sigma 0.05
```

输出：

- `results/baselines/baseline_comparison.json`

基线包含：`no_defense`、`gaussian_img`、`gaussian_emb`、`random_patch`、`recon_only`、`full_pipeline`。

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
