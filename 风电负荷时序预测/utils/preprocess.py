# -*- coding: utf-8 -*-
"""
原始风机数据处理脚本
将原始 CSV 数据处理为本模型可用的格式（3维特征，约1年数据）
"""

import pandas as pd
import numpy as np
import os

# =====================
# 1. 读取原始数据
# =====================
print("[1/4] 读取原始数据...")
data_path = os.path.join(os.path.dirname(__file__), '..', 'out_data', 'Turbine_Data.csv')
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
print(f"  原始数据量: {len(df)} 行, {len(df.columns)} 列")
print(f"  时间范围: {df.index.min()} ~ {df.index.max()}")

# =====================
# 2. 筛选1年数据（2019年）
# =====================
print("\n[2/4] 筛选2019全年数据...")
df_2019 = df.loc['2019'].copy()
print(f"  2019年数据量: {len(df_2019)} 行")
print(f"  时间范围: {df_2019.index.min()} ~ {df_2019.index.max()}")

# =====================
# 3. 选取3个核心特征
# =====================
print("\n[3/4] 选取核心特征...")
# 选择: ActivePower(有功功率/负荷), WindSpeed(风速), AmbientTemperatue(环境温度)
selected_cols = ['ActivePower', 'WindSpeed', 'AmbientTemperatue']
df_selected = df_2019[selected_cols].copy()

# 列名改为中文以便模型使用
df_selected.columns = ['ActivePower', 'WindSpeed', 'Temperature']

print(f"  选取特征: {list(df_selected.columns)}")
print(f"  缺失值统计:")
for col in df_selected.columns:
    missing_count = df_selected[col].isna().sum()
    print(f"    {col}: {missing_count}/{len(df_selected)} ({missing_count/len(df_selected)*100:.1f}%)")

# =====================
# 4. 处理缺失值 & 重采样
# =====================
print("\n[4/4] 处理缺失值与重采样...")

# 4a. 线性插值填补缺失值（时序数据适用）
df_filled = df_selected.interpolate(method='linear', limit_direction='both')

# 4b. 按小时重采样，取均值（减少数据量，同时平滑噪声）
df_hourly = df_filled.resample('1h').mean()

# 4c. 检查是否还有缺失
print(f"  插值+重采样后缺失值:")
for col in df_hourly.columns:
    missing_count = df_hourly[col].isna().sum()
    print(f"    {col}: {missing_count}")

# 如果重采样后还有极少量缺失，用前向填充
df_clean = df_hourly.ffill().bfill()

# =====================
# 5. 保存处理后的数据
# =====================
output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_2019.csv')
df_clean.to_csv(output_path, float_format='%.4f')

# 同时保存为numpy格式，方便模型直接加载
output_npy = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_2019.npy')
np.save(output_npy, df_clean.values)

print(f"\n✅ 处理完成!")
print(f"  输出文件: {output_path}")
print(f"  输出文件: {output_npy}")
print(f"  数据形状: {df_clean.values.shape} (时间步数={len(df_clean)}, 特征数=3)")
print(f"  时间范围: {df_clean.index.min()} ~ {df_clean.index.max()}")
print(f"  采样频率: 1小时")
print(f"  总时间步数: {len(df_clean)} (约 {len(df_clean)/24:.0f} 天)")
print(f"\n数据统计:")
print(f"  ActivePower  - 均值: {df_clean['ActivePower'].mean():.2f}, 标准差: {df_clean['ActivePower'].std():.2f}")
print(f"  WindSpeed    - 均值: {df_clean['WindSpeed'].mean():.2f}, 标准差: {df_clean['WindSpeed'].std():.2f}")
print(f"  Temperature  - 均值: {df_clean['Temperature'].mean():.2f}, 标准差: {df_clean['Temperature'].std():.2f}")
