# -*- coding: utf-8 -*-
"""
数据加载和生成模块
用于生成或加载风电负荷时序数据
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


class WindLoadDataset(Dataset):
    """
    风电负荷数据集类
    支持3维输入：(时间步, 特征维度) 其中特征包括 [负荷, 风速, 温度]
    """
    
    def __init__(self, data, sequence_length=24):
        """
        初始化数据集
        
        Args:
            data: numpy数组，形状为 (时间步数, 特征数)
            sequence_length: 时间序列长度（用多少个时间步预测下一个）
        """
        self.data = data  # 形状: (N, 3)
        self.sequence_length = sequence_length
        self.samples = []
        self.targets = []
        
        # 构建时序样本
        for i in range(len(data) - sequence_length):
            self.samples.append(data[i:i + sequence_length])  # 输入序列: (sequence_length, 3)
            self.targets.append(data[i + sequence_length, 0])  # 目标: 下一时刻的负荷值
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        返回单个样本
        
        Returns:
            sample: (sequence_length, 3)
            target: 标量
        """
        sample = torch.FloatTensor(self.samples[idx])
        target = torch.FloatTensor([self.targets[idx]])
        return sample, target


def generate_synthetic_data(num_samples=1000, num_features=3):
    """
    生成合成的风电负荷时序数据
    包含3个特征：[风电负荷, 风速, 温度]
    
    Args:
        num_samples: 生成样本数
        num_features: 特征维度 (固定为3)
    
    Returns:
        data: numpy数组，形状为 (num_samples, 3)
    """
    # 设置随机种子确保可重复
    np.random.seed(42)
    
    # 基础趋势（模拟不同时间的风电负荷）
    t = np.arange(num_samples)
    
    # 特征1：风电负荷 (主要目标)
    # 由日周期 + 随机波动 + 趋势组成
    load = (
        100 + 50 * np.sin(2 * np.pi * t / 24) +  # 日周期
        20 * np.sin(2 * np.pi * t / 168) +  # 周周期
        0.1 * t +  # 长期趋势
        np.random.normal(0, 10, num_samples)  # 随机噪声
    )
    
    # 特征2：风速（会影响风电负荷）
    wind_speed = (
        10 + 5 * np.sin(2 * np.pi * t / 48) +
        3 * np.cos(2 * np.pi * t / 168) +
        np.random.normal(0, 2, num_samples)
    )
    
    # 特征3：温度（可能影响系统需求）
    temperature = (
        20 + 10 * np.sin(2 * np.pi * t / 24) +
        5 * np.sin(2 * np.pi * t / 168) +
        np.random.normal(0, 1, num_samples)
    )
    
    # 组合成数据矩阵
    data = np.column_stack([load, wind_speed, temperature])
    
    return data


def normalize_data(data, scaler=None):
    """
    归一化数据到0-1范围
    
    Args:
        data: numpy数组，形状为 (N, num_features)
        scaler: 已训练的StandardScaler对象（用于测试集）
    
    Returns:
        normalized_data: 归一化后的数据
        scaler: StandardScaler对象（用于后续反归一化）
    """
    if scaler is None:
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data)
    else:
        normalized_data = scaler.transform(data)
    
    return normalized_data, scaler


def split_data(data, train_ratio=0.7, val_ratio=0.15):
    """
    将数据分为训练集、验证集、测试集
    
    Args:
        data: 完整的时序数据
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    
    Returns:
        train_data, val_data, test_data: 分割后的数据
    """
    total_len = len(data)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    
    train_data = data[:train_len]
    val_data = data[train_len:train_len + val_len]
    test_data = data[train_len + val_len:]
    
    return train_data, val_data, test_data


def create_data_loaders(data, sequence_length=24, batch_size=32, train_ratio=0.7, val_ratio=0.15):
    """
    创建数据加载器
    
    Args:
        data: 原始数据
        sequence_length: 时序长度
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    
    Returns:
        train_loader, val_loader, test_loader: 数据加载器
        scaler: 数据归一化器
    """
    # 数据归一化
    normalized_data, scaler = normalize_data(data)
    
    # 数据分割
    train_data, val_data, test_data = split_data(normalized_data, train_ratio, val_ratio)
    
    # 创建数据集
    train_dataset = WindLoadDataset(train_data, sequence_length)
    val_dataset = WindLoadDataset(val_data, sequence_length)
    test_dataset = WindLoadDataset(test_data, sequence_length)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler


def load_real_data(data_dir='data', sequence_length=24, batch_size=32):
    """
    加载预处理并切分好的真实风机数据

    Args:
        data_dir: 数据目录路径
        sequence_length: 时序长度
        batch_size: 批次大小

    Returns:
        train_loader, val_loader, test_loader: 数据加载器
        scaler: 数据归一化器
    """
    import os

    # 加载预切分的 .npy 文件
    train_data = np.load(os.path.join(data_dir, 'train_data.npy'))
    val_data = np.load(os.path.join(data_dir, 'val_data.npy'))
    test_data = np.load(os.path.join(data_dir, 'test_data.npy'))

    print(f"  训练集: {train_data.shape[0]} 条")
    print(f"  验证集: {val_data.shape[0]} 条")
    print(f"  测试集: {test_data.shape[0]} 条")

    # 用训练集拟合归一化器，再统一归一化
    scaler = StandardScaler()
    train_normalized = scaler.fit_transform(train_data)
    val_normalized = scaler.transform(val_data)
    test_normalized = scaler.transform(test_data)

    # 创建数据集
    train_dataset = WindLoadDataset(train_normalized, sequence_length)
    val_dataset = WindLoadDataset(val_normalized, sequence_length)
    test_dataset = WindLoadDataset(test_normalized, sequence_length)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler
