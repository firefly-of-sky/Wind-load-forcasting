# -*- coding: utf-8 -*-
"""
LSTM神经网络模型定义
用于风电负荷时序预测
"""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    基于LSTM的风电负荷预测模型
    
    输入维度：(batch_size, sequence_length, 3)
    - 3个特征：负荷、风速、温度
    输出维度：(batch_size, 1)
    - 单步预测下一时刻的负荷值
    """
    
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        """
        初始化LSTM模型
        
        Args:
            input_size: 输入特征维度 (3: 负荷、风速、温度)
            hidden_size: LSTM隐藏单元数
            num_layers: LSTM层数
            dropout: Dropout比例
            output_size: 输出维度 (1: 单步预测)
        """
        super(LSTMForecaster, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM层
        # - batch_first=True 使输入格式为 (batch, seq_len, feature)
        # - dropout 在层之间应用（num_layers > 1时有效）
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层用于最终预测
        # LSTM输出是 (batch_size, seq_len, hidden_size)
        # 我们只用最后一个时刻的输出 (batch_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_size)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状 (batch_size, sequence_length, 3)
        
        Returns:
            output: 预测值，形状 (batch_size, 1)
        """
        # LSTM前向传播
        # lstm_out: (batch_size, seq_len, hidden_size)
        # hidden: 最终隐藏状态
        # cell: 最终细胞状态
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 取最后一个时刻的输出
        # lstm_out[:, -1, :] 的形状: (batch_size, hidden_size)
        last_output = lstm_out[:, -1, :]
        
        # 全连接层处理
        out = self.fc1(last_output)  # (batch_size, 32)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # (batch_size, 1)
        
        return out


class AttentionLSTMForecaster(nn.Module):
    """
    带注意力机制的LSTM预测模型
    注意力机制可以学习不同时刻对预测的重要性权重
    """
    
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        """
        初始化注意力LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏单元数
            num_layers: LSTM层数
            dropout: Dropout比例
            output_size: 输出维度
        """
        super(AttentionLSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力层
        # 计算每个时刻的注意力权重
        self.attention = nn.Linear(hidden_size, 1)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_size)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状 (batch_size, sequence_length, 3)
        
        Returns:
            output: 预测值，形状 (batch_size, 1)
        """
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size)
        
        # 注意力机制
        # 为每个时刻计算注意力权重
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)  # 沿时间维度进行softmax
        
        # 加权平均所有时刻的输出
        # (batch_size, seq_len, 1) * (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 全连接层处理
        out = self.fc1(attended_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
