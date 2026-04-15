# STEER 图像检索实验

基于 STEER 论文（隐私保护向量检索）的图像检索实验复现。将原论文中的文本查询场景迁移到图像检索场景。

## 核心思路

1. **服务端编码器** (`enc_s`): 大模型 (CLIP ViT-L/14)，用于构建图库向量索引
2. **客户端编码器** (`enc_l`): 小模型 (CLIP ViT-B/32)，本地提取查询图像特征
3. **映射器** (`φ`): 学习从客户端嵌入空间到服务端嵌入空间的映射
4. 客户端只发送映射后的向量给服务端检索，不暴露原始图像

## 目录结构

```
steer_image/
├── configs/
│   └── default.yaml           # 全局配置
├── models/
│   ├── __init__.py
│   ├── encoders.py            # 本地/服务端图像编码器 (OpenCLIP)
│   └── mappers.py             # 线性/MLP空间映射器
├── data/
│   ├── __init__.py
│   └── dataset.py             # 数据集加载与对齐集构建
├── scripts/
│   ├── extract_embeddings.py  # 提取本地+服务端嵌入
│   ├── train_mapper.py        # 训练映射器
│   ├── build_index.py         # 构建FAISS索引
│   ├── retrieve.py            # 检索 (单次/批量/oracle)
│   ├── evaluate.py            # 综合评估 (Recall/mAP/对齐质量)
│   └── run_pipeline.py        # 一键运行全流程
├── requirements.txt
└── README.md
```

## 安装依赖

```bash
pip install -r steer_image/requirements.txt
```

## 快速开始

### 一键运行全流程

```bash
python -m steer_image.scripts.run_pipeline --config steer_image/configs/default.yaml
```

### 分步运行

```bash
# 1. 提取嵌入
python -m steer_image.scripts.extract_embeddings --config steer_image/configs/default.yaml

# 2. 训练映射器 (MLP)
python -m steer_image.scripts.train_mapper --config steer_image/configs/default.yaml

# 2b. 或训练线性映射器
python -m steer_image.scripts.train_mapper --config steer_image/configs/default.yaml --mapper_type linear

# 3. 构建FAISS索引
python -m steer_image.scripts.build_index --config steer_image/configs/default.yaml

# 4. 综合评估
python -m steer_image.scripts.evaluate --config steer_image/configs/default.yaml
```

## 评估指标

### 检索指标
- **Recall@k** (k=1,5,10,20): top-k中至少有一个正确结果的查询比例
- **Precision@k**: top-k结果中正确的比例
- **mAP**: 平均精度均值

### 对齐质量指标
- **MSE**: 映射后向量与真实服务端向量的均方误差
- **Cosine Similarity**: 余弦相似度
- **Neighbor Consistency**: top-k邻居重叠率

### 对比实验
- **Mapper**: 使用映射器的检索结果
- **Oracle**: 直接使用服务端编码器的检索结果（性能上界）

## 配置说明

主要配置项在 `configs/default.yaml` 中：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `encoders.local.name` | 客户端模型 | ViT-B-32 |
| `encoders.server.name` | 服务端模型 | ViT-L-14 |
| `mapper.type` | 映射器类型 | mlp |
| `dataset.name` | 数据集 | cifar100 |
| `training.loss` | 损失函数 | mse+cosine |
| `training.num_epochs` | 训练轮数 | 50 |
