# -*- coding: utf-8 -*-
"""
模型训练和评估模块
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Trainer:
    """
    模型训练器
    负责训练、验证和评估模型
    """
    
    def __init__(self, model, device='cpu', learning_rate=0.001):
        """
        初始化训练器
        
        Args:
            model: PyTorch模型
            device: 计算设备 ('cpu' 或 'cuda')
            learning_rate: 学习率
        """
        self.model = model.to(device)
        self.device = device
        
        # 损失函数：使用均方误差损失
        self.criterion = nn.MSELoss()
        
        # 优化器：使用Adam优化器
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # 学习率调度器：当验证损失不改进时降低学习率
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader):
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
        
        Returns:
            avg_loss: 平均训练损失
        """
        self.model.train()  # 设置为训练模式
        total_loss = 0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            # 将数据移到指定设备
            X = X.to(self.device)
            y = y.to(self.device)
            
            # 前向传播
            predictions = self.model(X)
            
            # 计算损失
            loss = self.criterion(predictions, y)
            
            # 反向传播
            self.optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 计算梯度
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
        
        Returns:
            avg_loss: 平均验证损失
            predictions: 所有预测值
            targets: 所有目标值
        """
        self.model.eval()  # 设置为评估模式
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():  # 不计算梯度
            for X, y in val_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                predictions = self.model(X)
                loss = self.criterion(predictions, y)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(y.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss, np.array(all_predictions), np.array(all_targets)
    
    def train(self, train_loader, val_loader, epochs=100, early_stopping_patience=15):
        """
        完整的训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 最大训练轮数
            early_stopping_patience: 早停等待轮数
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"开始训练模型 (设备: {self.device})")
        print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'LR':<10}")
        print("-" * 40)
        
        for epoch in range(epochs):
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, _, _ = self.validate(val_loader)
            
            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 打印进度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"{epoch+1:<6} {train_loss:<12.6f} {val_loss:<12.6f} {current_lr:<10.6f}")
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最好的模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n早停触发: 在第 {epoch+1} 个epoch后停止训练")
                    # 加载最好的模型
                    self.model.load_state_dict(
                        torch.load('best_model.pth', map_location=self.device, weights_only=True)
                    )
                    break
        
        print("训练完成！")
    
    def evaluate(self, test_loader):
        """
        在测试集上评估模型
        
        Args:
            test_loader: 测试数据加载器
        
        Returns:
            metrics: 包含MSE、RMSE、MAE、R2等指标的字典
            predictions: 预测值
            targets: 目标值
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                predictions = self.model(X)
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(y.cpu().numpy().flatten())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # 计算评估指标
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        return metrics, predictions, targets
