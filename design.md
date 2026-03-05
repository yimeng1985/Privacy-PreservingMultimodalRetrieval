
# 图像 Embedding 抗反演隐私保护项目整体设计方案（Markdown 版）

## 0. 一句话概述

在“**客户端本地生成图像 embedding → 云端向量库检索**”的场景下，embedding 可能被攻击者用于重建/反演原始图像，从而导致隐私泄露。本项目设计一种 **客户端侧前置防御**：通过“**稀疏隐私区域选择 + 重建引导的局部扰动**”生成受保护 embedding，在尽量保留检索效用的同时显著降低反演重建质量。

---

## 1. 背景与问题定义

### 1.1 场景描述（系统视角）

* 运营商在云端提供公共向量数据库（向量索引/检索服务）。
* 客户端收到查询（图像）后，在本地使用编码器 `E(·)` 计算 embedding `z` 并上传。
* 云端根据 `z` 在向量库中执行 ANN 检索并返回结果。

### 1.2 风险点（隐私视角）

* 上传到云端的是 embedding 而非原图，但 embedding 仍可能泄露可逆信息。
* 攻击者若能拿到 embedding（云端服务、运营商、入侵者、恶意第三方等），可能训练反演模型 `R(·)`：
  `x_hat = R(z)`，并尽量逼近原图 `x`。

### 1.3 研究目标（你要解决的核心问题）

在 **不显著破坏检索效果** 的前提下，让攻击者从 embedding 反演原图更困难，具体目标是优化：

* **隐私性（Privacy）：** 重建质量下降（PSNR/SSIM↓、LPIPS↑等）。
* **效用（Utility）：** 向量检索性能保持（Recall@K/mAP/NDCG 等维持）。
* **可部署性（Deployability）：** 客户端可承受的额外开销（延迟/算力/能耗）。

---

## 2. 威胁模型与假设

### 2.1 攻击者能力

* **黑盒/半黑盒：** 攻击者可能不知道客户端防御细节，但可收集大量 `(embedding, image)` 训练数据，训练反演器。
* **白盒：** 攻击者知道编码器结构或可复现 `E`（常见于公开模型如 CLIP/ViT）。

### 2.2 防御者能力

* 防御在 **客户端本地** 执行，可控 `E`（可选：固定预训练模型或微调）。
* 防御者可训练 **影子反演模型（shadow reconstructor）** 来近似攻击者能力，辅助生成扰动。

### 2.3 你的基本立场

* 本项目优先做 **经验型抗反演防御**（强实践效果、强可部署性）。
* 不追求严格差分隐私的数学证明（后续可做混合策略扩展）。

---

## 3. 设计目标与关键约束

### 3.1 目标

1. **最大化反演难度**：让重建器 `R(z')` 输出更模糊、更不可用、更难提取敏感信息。
2. **最小化 embedding 漂移**：`z'` 与 `z` 不应偏离太大，保证检索召回。
3. **稀疏/局部扰动**：只动少数“隐私承载区域”，更利于 tradeoff 与解释性。
4. **工程可实现**：能在你后续逐步实现的路线下收敛出稳定结果。

### 3.2 约束（典型部署约束）

* 客户端延迟预算：例如 20–100ms（取决于端侧设备）
* 额外显存/内存占用
* 不能依赖云端参与（云端是公共服务，不可改）

---

## 4. 总体方案：SPAG（Sparse Privacy-region selection + Adversarial perturbation for image embeddinGs）

### 4.1 框架概览

输入图像 `x`
→ 编码器 `E` 得到 `z = E(x)`
→ 选择器 `G` 找到隐私关键区域（patch / block）形成 mask `M`
→ 扰动生成器 `P` 仅在 `M` 指示的区域内加入扰动 `δ` 得到 `x' = x + M ⊙ δ`
→ 输出受保护 embedding：`z' = E(x')`

最终上传 `z'` 到云端检索。

---

## 5. 核心思想升级：块的重要性不是“embedding 变化大”，而是“隐私收益 / 语义代价”大

你原来的直觉（遮掉某块看 embedding 变化）是合理起点，但需要升级为双指标：

### 5.1 语义代价（Utility Cost）

遮掉第 `j` 个块/patch 后，embedding 与原 embedding 的变化：

* `z = E(x)`
* `x_{-j}`：对第 `j` 块做遮蔽/模糊/均值替换后的图
* `u_j = 1 - cos(E(x), E(x_{-j}))`

`u_j` 越大 → 该块对检索语义越关键 → 不宜动。

### 5.2 隐私收益（Privacy Gain）

遮掉第 `j` 块后，影子反演模型的重建损失增加量：

* `L_rec(·)`：重建损失（L1 + LPIPS 等）
* `p_j = L_rec(R(E(x_{-j})), x) - L_rec(R(E(x)), x)`

`p_j` 越大 → 该块对反演重建越关键 → 优先保护。

### 5.3 最终块评分（推荐）

用“隐私收益/语义代价”做排序：

* `s_j = p_j / (u_j + ε)`
  选择 top-k 个块或在总 `u_j` 预算下选块。

> 直观解释：优先动“**让重建变差很多，但对检索影响相对小**”的区域。

---

## 6. 模块化设计与模型选型

### 6.1 编码器 `E`（客户端本地）

**推荐优先使用 ViT 类编码器**（与 patch 思路天然契合）

可选：

* `CLIP ViT-B/16`（强基线，通用检索表现好）
* 轻量端侧：ViT-S/16 或更小模型（部署导向）

> 第一阶段建议固定 `E`，不要先微调；待防御机制稳定后再考虑端到端微调。

---

### 6.2 影子反演模型 `R`（防御端训练，用于指导扰动）

目标：逼近攻击者可能训练的 `R`，用于提供“让重建更差”的梯度信号。

推荐：

* **至少 1 个解码器**起步（ResDecoder 或 U-Net decoder）
* 后续升级为 **小型集成（ensemble）**：`R1, R2, R3`
  增强迁移性与稳健性（攻击者结构未知）

训练数据：公开数据集（与你业务域越接近越好）

训练目标：

* `x_hat = R(E(x))`
* `L_shadow = λ1 * L1(x_hat, x) + λ2 * LPIPS(x_hat, x)`（可先不用 GAN，稳定优先）

---

### 6.3 隐私区域选择器 `G`（两阶段实现）

#### 阶段 1：非参数 occlusion 打分（建议你先做这个）

* 分块：像素块（8/16/32）或 ViT patch（16×16）
* 对每块做遮蔽生成 `x_{-j}`
* 计算 `u_j, p_j, s_j`
* 选 top-k 块得到 mask `M`

优点：实现快、验证快、可解释。

#### 阶段 2：可学习稀疏 gate（升级版）

* 输入：patch tokens 或中间特征
* 输出：每个 patch 的 gate `m_j ∈ [0,1]`
* 加稀疏正则（近似 L0，如 hard-concrete / Gumbel 等）
* 训练目标：最少 patch 改动 → 最大重建恶化 + 最小语义损失

---

### 6.4 扰动生成器 `P`（局部、重建引导、可控预算）

核心：仅对选中区域 `M` 做 **masked PGD**（或其它可控优化）：

* `x' = clip(x + M ⊙ δ, 0, 1)`
* 约束：`||M ⊙ δ||_∞ ≤ ε_p` 或 `||M ⊙ δ||_2 ≤ ε_p`

优化目标（最大化隐私、最小化效用损失）：

* 隐私项（让重建更差）：
  `L_priv = Σ_m w_m * L_rec(R_m(E(x')), x)`
* 效用项（保持 embedding）：
  `L_util = 1 - cos(E(x'), E(x))`
* 平滑/视觉正则（可选）：
  `L_smooth = TV(M ⊙ δ) + ||M ⊙ δ||_2`

总体优化（最小化）：

* `L_total = -L_priv + λ_u L_util + λ_s L_sparse + λ_v L_smooth`

---

## 7. 算法流程（MVP 版）

### 7.1 输入输出

* 输入：图像 `x`
* 输出：受保护 embedding `z'`

### 7.2 流程（伪代码）

```text
Given image x, encoder E, shadow reconstructor R
1) z = E(x)

2) Split x into N patches {patch_j}

3) For each patch j:
     x_minus = Occlude(x, j)          # mask / blur / mean fill
     u_j = 1 - cos(E(x_minus), z)     # utility cost
     p_j = L_rec(R(E(x_minus)), x) - L_rec(R(z), x)   # privacy gain
     s_j = p_j / (u_j + eps)

4) Select top-k patches by s_j -> mask M

5) Initialize δ = 0
   For t = 1..T (PGD steps):
      x' = clip(x + M ⊙ δ)
      loss = -L_rec(R(E(x')), x) + λ_u*(1 - cos(E(x'), z)) + λ_v*TV(M ⊙ δ)
      δ = Project(δ - α*sign(∇_δ loss), ε_p)   # masked PGD

6) z' = E(x')
7) Upload z' to vector DB
```

---

## 8. 训练与实现路线（分阶段里程碑）

### Phase 0：准备与基线（1–2 周）

* 选定 `E`（建议 CLIP ViT-B/16）
* 训练/复现 1 个基础反演器 `R`
* 建立 baseline：

  * 无防御：`z = E(x)` → 攻击重建质量
  * 简单噪声：`x + Gaussian(σ)` 或 `z + Gaussian(σ)`（作为弱对照）
  * 简单遮蔽：随机遮块（作为对照）

产出：baseline 指标表（privacy + utility + overhead）

---

### Phase 1：MVP 防御（2–4 周）

* 实现 occlusion scoring：`u_j, p_j, s_j`
* 实现 top-k patch mask `M`
* 实现 masked PGD 生成 `x'`
* 输出 `z'`，测试检索与重建指标

产出：

* 防御效果显著优于“随机遮块/全局噪声”
* 出第一版消融（块大小、k、ε、PGD step）

---

### Phase 2：稳健性与迁移性（4–8 周）

* 训练多个影子反演器（ensemble）
* 攻击侧换结构测试（迁移）
* 引入更贴近攻击者的设置（白盒/半白盒）

产出：对未知攻击结构更稳健的结果曲线

---

### Phase 3：学习型选择器（可选，8 周+）

* 将 `G` 从非参数升级到可学习稀疏 gate
* 支持 token-level 或 feature-level 选择
* 端到端优化：`G + P`（在效用约束下最大化隐私）

产出：更好的 privacy-utility tradeoff、更像论文创新点

---

## 9. 评估协议（必须完整）

### 9.1 隐私（反演攻击效果）

对多个攻击器评估（至少 3 个结构）：

* PSNR（↓）
* SSIM（↓）
* LPIPS（↑）
* MSE（↑）
* 若是人脸/身份敏感：额外做人脸识别/属性识别在重建图上的成功率

### 9.2 效用（检索效果）

* Recall@K、mAP、NDCG@K
* Top-K 邻居一致率（防御前后检索结果重合比例）
* embedding cosine drift（均值/分位数）

### 9.3 开销（部署）

* 端侧额外延迟（occlusion + PGD）
* 额外显存/内存
* 额外功耗（可选）

---

## 10. 消融实验清单

1. **块大小/粒度**：8×8 / 16×16 / 32×32 / ViT patch
2. **选择比例 k**：5%、10%、20%、30%、50%
3. **扰动方式**：遮蔽 vs 高斯噪声 vs masked PGD（你主方法）
4. **区域选择策略**：随机 / 只看 drift / 只看 privacy gain / ratio（你方法）/ 学习型 gate
5. **攻击迁移性**：不同 `R` 架构、不同训练数据、白盒/黑盒
6. **损失项贡献**：去掉 `L_util` / 去掉 `L_smooth` / 去掉 ensemble

---

## 11. 风险点与对策（提前规避）

### 风险 1：只动“embedding 变化大”的块会伤检索

* 对策：用 ratio `s_j = p_j/(u_j+ε)` 选块，而非只看 `u_j`

### 风险 2：shadow 反演器过弱导致扰动无效

* 对策：先确保 baseline 反演器能重建出可识别结构；再做防御
* 升级：ensemble 增强迁移性

### 风险 3：PGD 太慢不适合端侧

* 对策：减少迭代步（5–10）、只扰动少量块、或训练一个“单步生成器”（后续可选）

### 风险 4：对某些数据域有效，对另一些失效

* 对策：跨数据集评估；做 domain-specific 影子模型微调

---

## 12. 项目代码结构建议

```text
project/
  configs/
  data/
  models/
    encoder.py            # E
    reconstructor.py      # R variants
    selector.py           # occlusion scoring / gate model
    perturb.py            # masked PGD / noise
  attacks/
    inversion_attack.py   # attack training & eval
  eval/
    metrics_privacy.py    # psnr/ssim/lpips
    metrics_utility.py    # recall@k/map/ndcg
  scripts/
    train_shadow_R.py
    run_defense.py
    eval_attack.py
    ablation_grid.py
  README.md
```

---

## 13. 参考基础工作

* 文本 embedding 场景下：通过检测“隐私敏感神经元/维度”并进行定向扰动，提高反演难度（你的第一篇参考）
* 图像/人脸特征场景下：通过 shadow reconstructor 的重建损失梯度指导对抗扰动，提高反演难度（你的第二篇参考）
