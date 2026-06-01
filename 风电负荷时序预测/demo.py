# -*- coding: utf-8 -*-
"""
风电负荷时序预测 - 完整Demo
演示从真实数据加载到模型训练、评估的完整流程
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_real_data
from models.lstm_model import LSTMForecaster, AttentionLSTMForecaster
from utils.trainer import Trainer


def plot_results(train_losses, val_losses, test_predictions, test_targets):
    """
    绘制训练曲线和预测结果
    
    Args:
        train_losses: 训练损失历史
        val_losses: 验证损失历史
        test_predictions: 测试集预测值
        test_targets: 测试集目标值
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot loss curves
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot predictions vs ground truth
    time_steps = range(1, min(200, len(test_predictions)) + 1)
    axes[1].plot(time_steps, test_targets[:len(time_steps)], 'g-', label='Ground Truth', linewidth=2)
    axes[1].plot(time_steps, test_predictions[:len(time_steps)], 'r--', label='Prediction', linewidth=2)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Load (MW)')
    axes[1].set_title('Test Set Predictions (first 200 steps)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/training_results.png', dpi=100, bbox_inches='tight')
    print("Results chart saved to: results/training_results.png")
    plt.close()


def main():
    """
    主程序：完整的风电负荷预测pipeline
    """
    print("=" * 60)
    print("       风电负荷时序预测 - 深度学习演示")
    print("=" * 60)
    
    # =====================
    # 1. 数据加载
    # =====================
    print("\n[1/4] 加载真实风机数据...")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    sequence_length = 24  # 使用过去24小时预测下一小时
    batch_size = 32

    train_loader, val_loader, test_loader, scaler = load_real_data(
        data_dir=data_dir,
        sequence_length=sequence_length,
        batch_size=batch_size
    )

    print(f"✓ 特征: ActivePower(负荷), WindSpeed(风速), Temperature(温度)")
    print(f"✓ 时序长度: {sequence_length} 步")
    print(f"✓ 批次大小: {batch_size}")
    print(f"✓ 训练样本数: {len(train_loader.dataset)}")
    print(f"✓ 验证样本数: {len(val_loader.dataset)}")
    print(f"✓ 测试样本数: {len(test_loader.dataset)}")
    
    # =====================
    # 3. 模型构建
    # =====================
    print("\n[2/4] 构建模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ 使用设备: {device}")
    
    # 模型超参数
    input_size = 3  # 三个特征
    hidden_size = 64
    num_layers = 2
    dropout = 0.2
    
    # 创建LSTM模型
    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型参数总数: {total_params:,}")
    print("  架构:")
    print(f"    - LSTM: {num_layers}层, 隐藏单元数: {hidden_size}")
    print(f"    - 全连接层: {hidden_size} -> 32 -> 1")
    print(f"    - Dropout: {dropout}")
    
    # =====================
    # 4. 模型训练
    # =====================
    print("\n[3/4] 训练模型...")
    trainer = Trainer(model, device=device, learning_rate=0.001)
    
    trainer.train(
        train_loader,
        val_loader,
        epochs=50,
        early_stopping_patience=10
    )
    
    # =====================
    # 5. 模型评估
    # =====================
    print("\n[4/4] 评估模型...")
    metrics, predictions, targets = trainer.evaluate(test_loader)
    
    print("\n测试集性能指标:")
    print(f"  MSE  (均方误差)     : {metrics['MSE']:.6f}")
    print(f"  RMSE (均方根误差)   : {metrics['RMSE']:.6f}")
    print(f"  MAE  (平均绝对误差) : {metrics['MAE']:.6f}")
    print(f"  R²   (决定系数)     : {metrics['R2']:.6f}")
    
    # =====================
    # 结果可视化
    # =====================
    print("\n绘制结果...")
    plot_results(
        trainer.train_losses,
        trainer.val_losses,
        predictions,
        targets
    )
    
    # =====================
    # 保存模型
    # =====================
    print("\n保存模型...")
    model_path = 'results/wind_load_forecast_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"✓ 模型已保存到: {model_path}")
    
    print("\n" + "=" * 60)
    print("       演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
