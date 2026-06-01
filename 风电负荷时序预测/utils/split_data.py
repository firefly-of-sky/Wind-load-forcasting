# -*- coding: utf-8 -*-
"""
将处理后的数据按比例切分为训练集、验证集、测试集
"""

import numpy as np
import os

# =====================
# 1. 加载处理后的数据
# =====================
print("[1/3] 加载处理后的数据...")
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_2019.npy')
data = np.load(data_path)
print(f"  数据形状: {data.shape} (时间步={data.shape[0]}, 特征数={data.shape[1]})")

# =====================
# 2. 按比例切分 (时间序列顺序保留)
# =====================
print("\n[2/3] 按时间顺序切分数据...")
train_ratio = 0.7
val_ratio = 0.15

total_len = len(data)
train_len = int(total_len * train_ratio)
val_len = int(total_len * val_ratio)

train_data = data[:train_len]
val_data = data[train_len:train_len + val_len]
test_data = data[train_len + val_len:]

print(f"  训练集: {train_data.shape[0]} 条 ({train_len/total_len*100:.0f}%)")
print(f"  验证集: {val_data.shape[0]} 条 ({val_len/total_len*100:.0f}%)")
print(f"  测试集: {test_data.shape[0]} 条 ({(total_len-train_len-val_len)/total_len*100:.0f}%)")

# =====================
# 3. 保存
# =====================
print("\n[3/3] 保存切分数据...")
out_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

np.save(os.path.join(out_dir, 'train_data.npy'), train_data)
np.save(os.path.join(out_dir, 'val_data.npy'), val_data)
np.save(os.path.join(out_dir, 'test_data.npy'), test_data)

print(f"  ✓ train_data.npy ({train_data.shape})")
print(f"  ✓ val_data.npy ({val_data.shape})")
print(f"  ✓ test_data.npy ({test_data.shape})")
print("\n✅ 数据切分完成")
