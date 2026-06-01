# 风电负荷时序预测

基于深度学习的风电负荷时序预测系统，使用 PyTorch 构建。

## 项目概述

本系统使用 LSTM（长短期记忆）神经网络，结合多维输入特征预测风电负荷，输入特征包括：

- **风电负荷 (MW)** — 预测目标
- **风速 (m/s)**
- **温度 (°C)**

模型使用过去 24 个时间步的历史数据（三维输入），预测下一个时间步的负荷值。

## 项目结构

```
风电负荷时序预测/
├── demo.py                  # 主程序：数据生成 → 模型训练 → 评估
├── models/
│   └── lstm_model.py        # LSTM 与带注意力机制的 LSTM 模型定义
├── utils/
│   ├── data_loader.py       # 合成数据生成与 PyTorch DataLoader
│   └── trainer.py           # 训练循环、验证与评估
├── data/                    # 数据目录（放置真实数据集）
├── results/                 # 输出：训练好的模型与结果图
├── README.md
└── environment.yml          # Conda 环境配置文件
```

## 功能特性

- **多维度输入**：3 维及以上特征（负荷、风速、温度）
- **两种模型变体**：
  - `LSTMForecaster` — 标准堆叠 LSTM + 全连接输出层
  - `AttentionLSTMForecaster` — LSTM + 注意力机制，为不同时间步分配权重
- **训练工具**：
  - Adam 优化器 + ReduceLROnPlateau 学习率调度
  - 梯度裁剪，防止梯度爆炸
  - 早停机制 + 最佳模型自动保存
- **评估指标**：MSE、RMSE、MAE、R²
- **可视化**：损失曲线 + 预测值 vs 真实值对比图
- **GPU 加速**：支持 CUDA（兼容 NVIDIA RTX 5060 / Blackwell 架构）

## 环境要求

| 依赖库 | 版本 |
|--------|------|
| Python | 3.10 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0 |
| torchaudio | 2.11.0 |
| NumPy | ≥1.23.0 |
| Pandas | ≥1.5.0 |
| Matplotlib | ≥3.7.0 |
| Seaborn | ≥0.12.0 |
| Scikit-learn | ≥1.2.0 |
| SciPy | ≥1.10.0 |
| Statsmodels | ≥0.14.0 |
| Jupyter / JupyterLab | ≥3.6.0 |

## 环境配置

### 1. 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate wind-load-forecasting
```

### 2. NVIDIA RTX 50 系列 (Blackwell) GPU 用户须知

`environment.yml` 中的默认 PyTorch 版本可能不支持 Blackwell 架构 (sm_120) 的新显卡，需手动升级：

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. 验证安装

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 使用方法

运行完整的 Demo 流程：

```bash
cd 风电负荷时序预测
mkdir -p results
PYTHONIOENCODING=utf-8 python demo.py
```

Demo 执行流程：

1. 生成 1000 条合成数据（3 个特征）
2. 划分训练集 / 验证集 / 测试集（70% / 15% / 15%）
3. 构建 2 层 LSTM 模型（64 个隐藏单元）
4. 训练模型（最多 50 轮，含早停）
5. 在测试集上评估模型
6. 保存模型与结果图到 `results/` 目录

### 预期输出

```
============================================================
       风电负荷时序预测 - 深度学习演示
============================================================

[1/5] 生成合成数据...
[2/5] 准备数据加载器...
[3/5] 构建模型...
[4/5] 训练模型...
[5/5] 评估模型...

测试集性能指标:
  MSE  (均方误差)     : 0.0934
  RMSE (均方根误差)   : 0.3057
  MAE  (平均绝对误差) : 0.2552
  R²   (决定系数)     : 0.8550

绘制结果...
Results chart saved to: results/training_results.png

保存模型...
✓ 模型已保存到: results/wind_load_forecast_model.pth

============================================================
       演示完成!
============================================================
```

## 模型架构

```
输入: (batch_size, 24, 3)
    ↓
LSTM (输入维度=3, 隐藏单元=64, 层数=2, dropout=0.2)
    ↓
取最后一个时间步的输出: (batch_size, 64)
    ↓
Linear(64 → 32) → ReLU → Dropout(0.2)
    ↓
Linear(32 → 1)
    ↓
输出: (batch_size, 1)  ← 预测的负荷值
```

模型总参数量约 53,000。

## 训练参数配置

| 参数 | 取值 |
|------|------|
| 序列长度 | 24 个时间步 |
| 批次大小 | 32 |
| 优化器 | Adam (lr=0.001) |
| 损失函数 | MSE |
| 学习率调度 | ReduceLROnPlateau (factor=0.5, patience=5) |
| 早停耐心值 | 15 个 epoch |
| 梯度裁剪 | max_norm=1.0 |

## 工作原则

1. **规划优先**：编码前制定详细计划并提交审核
2. **遵守规范**：代码必须包含详细注释，结构清晰，模块逻辑分明
3. **遇事咨询**：遇到决策问题时，先询问用户再行动
4. **优化先审**：如有优化建议，先提出方案供用户审核，获批后再实施
