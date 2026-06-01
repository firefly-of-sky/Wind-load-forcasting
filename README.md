# 风电负荷时序预测

基于深度学习的风电负荷时序预测系统，使用 PyTorch 构建。本项目由 [Claude Code](https://claude.ai/code) 载入 DeepSeek-V4 模型辅助开发。

## 项目概述

本系统使用 LSTM（长短期记忆）神经网络，结合多维输入特征预测风电负荷，输入特征包括：

- **ActivePower (MW)** — 预测目标，风机有功功率/负荷
- **WindSpeed (m/s)** — 风速
- **Temperature (°C)** — 环境温度

模型使用过去 24 个时间步的历史数据（三维输入），预测下一个时间步的负荷值。

**数据来源**: [Wind Power Forecasting - Kaggle](https://www.kaggle.com/datasets/theforcecoder/wind-power-forecasting?resource=download)

## 项目结构

```
风电负荷时序预测/
├── demo.py                  # 主程序：加载真实数据 → 模型训练 → 评估
├── models/
│   └── lstm_model.py        # LSTM 与带注意力机制的 LSTM 模型定义
├── utils/
│   ├── data_loader.py       # 数据加载器（合成数据生成 + 真实数据加载）
│   ├── trainer.py           # 训练循环、验证与评估
│   ├── preprocess.py        # 原始数据预处理脚本（10min → 1h 重采样）
│   └── split_data.py        # 数据按比例切分脚本（70/15/15）
├── out_data/
│   └── Turbine_Data.csv     # 原始风机数据（从网上下载，10分钟间隔）
├── data/
│   ├── processed_2019.csv   # 处理后的2019全年数据 (8760行×3列)
│   ├── processed_2019.npy   # NumPy 格式
│   ├── train_data.npy       # 训练集 (70%)
│   ├── val_data.npy         # 验证集 (15%)
│   └── test_data.npy        # 测试集 (15%)
├── results/                 # 输出：训练好的模型与结果图
├── README.md
└── environment.yml          # Conda 环境配置文件
```

## 功能特性

- **多维度输入**：3 维及以上特征（负荷、风速、温度）
- **两种模型变体**：
  - `LSTMForecaster` — 标准堆叠 LSTM + 全连接输出层
  - `AttentionLSTMForecaster` — LSTM + 注意力机制，为不同时间步分配权重
- **真实数据支持**：
  - 原始 10 分钟间隔风机数据 → 插值填补 → 1 小时重采样
  - 自动按时间顺序切分训练/验证/测试集
- **训练工具**：
  - Adam 优化器 + ReduceLROnPlateau 学习率调度
  - 梯度裁剪，防止梯度爆炸
  - 早停机制 + 最佳模型自动保存
- **评估指标**：MSE、RMSE、MAE、R²
- **可视化**：损失曲线 + 预测值 vs 真实值对比图
- **GPU 加速**：支持 CUDA（兼容 NVIDIA RTX 5060 / Blackwell 架构）

## 开发环境

本项目使用 **Claude Code** 命令行工具，载入 **DeepSeek-V4** 大语言模型进行辅助开发，包括代码编写、环境配置、问题排查等全流程。

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

### 数据预处理（首次使用）

```bash
cd 风电负荷时序预测/utils

# Step 1: 原始数据 → 1年数据 → 3特征 → 1小时重采样
python preprocess.py

# Step 2: 按时间顺序切分训练/验证/测试集
python split_data.py
```

### 运行 Demo

```bash
cd 风电负荷时序预测
mkdir -p results
PYTHONIOENCODING=utf-8 python demo.py
```

Demo 执行流程：

1. 加载预处理并切分好的真实风机数据（8760 小时，2019 全年）
2. 构建 2 层 LSTM 模型（64 个隐藏单元）
3. 训练模型（最多 50 轮，含早停）
4. 在测试集上评估模型，保存模型与结果图

### 数据预处理详情

| 步骤 | 说明 |
|------|------|
| 原始数据 | 118,224 行 × 21 列，2017-12-31 ~ 2020-03-30，10 分钟间隔 |
| 筛选年份 | 2019 全年，52,560 行 |
| 选取特征 | `ActivePower`、`WindSpeed`、`AmbientTemperatue` |
| 缺失值处理 | 线性插值 + 1 小时均值重采样 |
| 最终数据 | 8,760 行 × 3 列，2019-01-01 ~ 2019-12-31，1 小时间隔 |
| 数据切分 | 训练集 70%（1~8月） / 验证集 15%（9~10月） / 测试集 15%（11~12月） |

### 实际输出

```
============================================================
       风电负荷时序预测 - 深度学习演示
============================================================

[1/4] 加载真实风机数据...
  训练集: 6132 条
  验证集: 1314 条
  测试集: 1314 条
✓ 特征: ActivePower(负荷), WindSpeed(风速), Temperature(温度)
✓ 时序长度: 24 步
✓ 批次大小: 32
✓ 训练样本数: 6108
✓ 验证样本数: 1290
✓ 测试样本数: 1290

[2/4] 构建模型...
✓ 使用设备: cuda
✓ 模型参数总数: 53,057

[3/4] 训练模型...
开始训练模型 (设备: cuda)
早停触发: 在第 13 个epoch后停止训练

[4/4] 评估模型...

测试集性能指标:
  MSE  (均方误差)     : 0.0635
  RMSE (均方根误差)   : 0.2521
  MAE  (平均绝对误差) : 0.1838
  R²   (决定系数)     : 0.7811
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
| 序列长度 | 24 个时间步（过去 24 小时预测下 1 小时） |
| 批次大小 | 32 |
| 优化器 | Adam (lr=0.001) |
| 损失函数 | MSE |
| 学习率调度 | ReduceLROnPlateau (factor=0.5, patience=5) |
| 早停耐心值 | 10 个 epoch |
| 梯度裁剪 | max_norm=1.0 |

## 工作原则

1. **规划优先**：编码前制定详细计划并提交审核
2. **遵守规范**：代码必须包含详细注释，结构清晰，模块逻辑分明
3. **遇事咨询**：遇到决策问题时，先询问用户再行动
4. **优化先审**：如有优化建议，先提出方案供用户审核，获批后再实施
