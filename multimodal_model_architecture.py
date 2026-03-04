#!/usr/bin/env python3
"""
多模态免疫表位预测模型架构
========================

功能：
1. 设计双分支多模态神经网络
2. 处理序列特征和结构特征
3. 支持缺失结构信息的情况
4. 实现B细胞/T细胞联合预测
5. 集成注意力机制和特征融合

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SequenceEncoder(nn.Module):
    """序列特征编码器"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.3):
        super(SequenceEncoder, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = current_dim
        
    def forward(self, x):
        return self.encoder(x)


class StructureEncoder(nn.Module):
    """结构特征编码器"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.3):
        super(StructureEncoder, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = current_dim
        
    def forward(self, x):
        return self.encoder(x)


class AttentionFusion(nn.Module):
    """注意力融合模块"""
    
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, seq_features, struct_features, has_structure_mask):
        # 为没有结构信息的样本使用零向量
        batch_size = seq_features.size(0)
        feature_dim = seq_features.size(1)
        
        # 创建结构特征的掩码版本
        masked_struct_features = struct_features * has_structure_mask.unsqueeze(1).float()
        
        # 连接特征
        combined_features = torch.cat([seq_features, masked_struct_features], dim=1)
        
        # 计算注意力权重
        attention_weights = self.attention(combined_features)
        
        # 应用注意力权重
        attended_seq = seq_features * attention_weights[:, 0:1]
        attended_struct = masked_struct_features * attention_weights[:, 1:2]
        
        # 融合特征
        fused_features = attended_seq + attended_struct
        
        return fused_features, attention_weights


class MultiModalEpitopePredictor(nn.Module):
    """多模态表位预测器"""
    
    def __init__(self, seq_input_dim, struct_input_dim, hidden_dims=[64, 32], 
                 num_classes=2, dropout=0.3):
        super(MultiModalEpitopePredictor, self).__init__()
        
        # 特征编码器
        self.sequence_encoder = SequenceEncoder(seq_input_dim, hidden_dims, dropout)
        self.structure_encoder = StructureEncoder(struct_input_dim, hidden_dims, dropout)
        
        # 确保两个编码器输出相同维度
        assert self.sequence_encoder.output_dim == self.structure_encoder.output_dim
        feature_dim = self.sequence_encoder.output_dim
        
        # 注意力融合
        self.attention_fusion = AttentionFusion(feature_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        # 辅助分类器 (用于多任务学习)
        self.epitope_type_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 2)  # B细胞 vs T细胞
        )
        
    def forward(self, seq_features, struct_features, has_structure_mask):
        # 编码特征
        seq_encoded = self.sequence_encoder(seq_features)
        struct_encoded = self.structure_encoder(struct_features)
        
        # 注意力融合
        fused_features, attention_weights = self.attention_fusion(
            seq_encoded, struct_encoded, has_structure_mask
        )
        
        # 预测
        main_output = self.classifier(fused_features)
        epitope_type_output = self.epitope_type_classifier(fused_features)
        
        return {
            'main_prediction': main_output,
            'epitope_type_prediction': epitope_type_output,
            'attention_weights': attention_weights,
            'fused_features': fused_features
        }


class EpitopeDataLoader:
    """表位数据加载器"""
    
    def __init__(self, data_path="data/structural_features/multimodal_feature_vectors.csv"):
        self.data_path = Path(data_path)
        self.scaler_seq = StandardScaler()
        self.scaler_struct = StandardScaler()
        
    def load_and_preprocess_data(self):
        """加载并预处理数据"""
        print("📊 加载多模态特征数据...")
        
        # 加载数据
        df = pd.read_csv(self.data_path)
        
        # 分离特征
        sequence_features = ['length', 'hydrophobicity', 'net_charge']
        structure_features = [
            'sasa_mean', 'sasa_std', 'sasa_max',
            'curvature_mean', 'curvature_std',
            'depth_mean', 'depth_std',
            'electrostatic_mean', 'electrostatic_std',
            'hydrophobic_mean', 'hydrophobic_std'
        ]
        
        # 提取特征矩阵
        X_seq = df[sequence_features].values
        X_struct = df[structure_features].values
        
        # 结构信息掩码
        has_structure = df['has_structure'].values.astype(bool)
        
        # 表位类型标签 (B细胞=0, T细胞=1)
        epitope_type_labels = (df['epitope_type'] == 'tcell').astype(int).values
        
        # 创建免疫原性标签 (这里我们假设所有表位都是免疫原性的，实际应用中需要真实标签)
        # 为了演示，我们基于一些特征创建合成标签
        immunogenicity_labels = self._create_synthetic_labels(df)
        
        # 标准化特征
        X_seq_scaled = self.scaler_seq.fit_transform(X_seq)
        X_struct_scaled = self.scaler_struct.fit_transform(X_struct)
        
        print(f"✅ 数据加载完成:")
        print(f"   📝 样本数: {len(df)}")
        print(f"   🧬 序列特征: {len(sequence_features)}个")
        print(f"   🏗️  结构特征: {len(structure_features)}个")
        print(f"   📊 有结构信息: {has_structure.sum()}个")
        
        return {
            'X_seq': X_seq_scaled,
            'X_struct': X_struct_scaled,
            'has_structure': has_structure,
            'epitope_type_labels': epitope_type_labels,
            'immunogenicity_labels': immunogenicity_labels,
            'sequences': df['sequence'].values
        }
    
    def _create_synthetic_labels(self, df):
        """创建合成的免疫原性标签"""
        # 基于疏水性、长度和表面可及性创建合成标签
        # 这只是为了演示，实际应用需要真实的实验标签
        
        scores = (
            df['hydrophobicity'].fillna(0) * 0.3 +
            (df['length'] - df['length'].mean()) / df['length'].std() * 0.2 +
            df['sasa_mean'].fillna(0) * 0.5
        )
        
        # 将得分转换为二分类标签
        threshold = scores.median()
        labels = (scores > threshold).astype(int).values
        
        return labels
    
    def create_data_splits(self, data, test_size=0.2, val_size=0.1, random_state=42):
        """创建训练/验证/测试集分割"""
        
        # 先分离训练+验证 和 测试集
        indices = np.arange(len(data['X_seq']))
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state,
            stratify=data['epitope_type_labels']
        )
        
        # 再分离训练和验证集
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_size/(1-test_size), random_state=random_state,
            stratify=data['epitope_type_labels'][train_val_idx]
        )
        
        def extract_subset(indices):
            return {
                'X_seq': torch.FloatTensor(data['X_seq'][indices]),
                'X_struct': torch.FloatTensor(data['X_struct'][indices]),
                'has_structure': torch.BoolTensor(data['has_structure'][indices]),
                'epitope_type_labels': torch.LongTensor(data['epitope_type_labels'][indices]),
                'immunogenicity_labels': torch.LongTensor(data['immunogenicity_labels'][indices]),
                'sequences': data['sequences'][indices]
            }
        
        splits = {
            'train': extract_subset(train_idx),
            'val': extract_subset(val_idx),
            'test': extract_subset(test_idx)
        }
        
        print(f"📊 数据分割完成:")
        print(f"   🏋️  训练集: {len(train_idx)}个样本")
        print(f"   ✅ 验证集: {len(val_idx)}个样本")
        print(f"   🧪 测试集: {len(test_idx)}个样本")
        
        return splits


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        
    def train_epoch(self, dataloader, optimizer, criterion_main, criterion_aux, alpha=0.7):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct_main = 0
        correct_aux = 0
        total_samples = 0
        
        for batch in dataloader:
            # 移动数据到设备
            X_seq = batch['X_seq'].to(self.device)
            X_struct = batch['X_struct'].to(self.device)
            has_structure = batch['has_structure'].to(self.device)
            main_labels = batch['immunogenicity_labels'].to(self.device)
            aux_labels = batch['epitope_type_labels'].to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = self.model(X_seq, X_struct, has_structure)
            
            # 计算损失
            main_loss = criterion_main(outputs['main_prediction'], main_labels)
            aux_loss = criterion_aux(outputs['epitope_type_prediction'], aux_labels)
            total_loss_batch = alpha * main_loss + (1 - alpha) * aux_loss
            
            # 反向传播
            total_loss_batch.backward()
            optimizer.step()
            
            # 统计准确率
            total_loss += total_loss_batch.item()
            
            _, predicted_main = torch.max(outputs['main_prediction'], 1)
            _, predicted_aux = torch.max(outputs['epitope_type_prediction'], 1)
            
            correct_main += (predicted_main == main_labels).sum().item()
            correct_aux += (predicted_aux == aux_labels).sum().item()
            total_samples += main_labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy_main = correct_main / total_samples
        accuracy_aux = correct_aux / total_samples
        
        return avg_loss, accuracy_main, accuracy_aux
    
    def evaluate(self, dataloader, criterion_main, criterion_aux, alpha=0.7):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct_main = 0
        correct_aux = 0
        total_samples = 0
        
        all_predictions_main = []
        all_labels_main = []
        all_predictions_aux = []
        all_labels_aux = []
        
        with torch.no_grad():
            for batch in dataloader:
                # 移动数据到设备
                X_seq = batch['X_seq'].to(self.device)
                X_struct = batch['X_struct'].to(self.device)
                has_structure = batch['has_structure'].to(self.device)
                main_labels = batch['immunogenicity_labels'].to(self.device)
                aux_labels = batch['epitope_type_labels'].to(self.device)
                
                # 前向传播
                outputs = self.model(X_seq, X_struct, has_structure)
                
                # 计算损失
                main_loss = criterion_main(outputs['main_prediction'], main_labels)
                aux_loss = criterion_aux(outputs['epitope_type_prediction'], aux_labels)
                total_loss_batch = alpha * main_loss + (1 - alpha) * aux_loss
                
                total_loss += total_loss_batch.item()
                
                # 预测
                _, predicted_main = torch.max(outputs['main_prediction'], 1)
                _, predicted_aux = torch.max(outputs['epitope_type_prediction'], 1)
                
                correct_main += (predicted_main == main_labels).sum().item()
                correct_aux += (predicted_aux == aux_labels).sum().item()
                total_samples += main_labels.size(0)
                
                # 收集预测结果
                all_predictions_main.extend(predicted_main.cpu().numpy())
                all_labels_main.extend(main_labels.cpu().numpy())
                all_predictions_aux.extend(predicted_aux.cpu().numpy())
                all_labels_aux.extend(aux_labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy_main = correct_main / total_samples
        accuracy_aux = correct_aux / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy_main': accuracy_main,
            'accuracy_aux': accuracy_aux,
            'predictions_main': all_predictions_main,
            'labels_main': all_labels_main,
            'predictions_aux': all_predictions_aux,
            'labels_aux': all_labels_aux
        }


def create_data_loaders(data_splits, batch_size=32):
    """创建数据加载器"""
    from torch.utils.data import DataLoader, TensorDataset
    
    def create_dataset(split_data):
        return TensorDataset(
            split_data['X_seq'],
            split_data['X_struct'],
            split_data['has_structure'],
            split_data['immunogenicity_labels'],
            split_data['epitope_type_labels']
        )
    
    # 自定义collate函数
    def collate_fn(batch):
        X_seq, X_struct, has_structure, immuno_labels, epitope_labels = zip(*batch)
        
        return {
            'X_seq': torch.stack(X_seq),
            'X_struct': torch.stack(X_struct),
            'has_structure': torch.stack(has_structure),
            'immunogenicity_labels': torch.stack(immuno_labels),
            'epitope_type_labels': torch.stack(epitope_labels)
        }
    
    loaders = {}
    for split_name, split_data in data_splits.items():
        dataset = create_dataset(split_data)
        shuffle = (split_name == 'train')
        
        loaders[split_name] = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=collate_fn
        )
    
    return loaders


def main():
    """主函数 - 演示模型架构"""
    print("🚀 启动多模态表位预测模型")
    print("="*60)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # 加载数据
    data_loader = EpitopeDataLoader()
    data = data_loader.load_and_preprocess_data()
    
    # 创建数据分割
    data_splits = data_loader.create_data_splits(data)
    
    # 创建模型
    seq_input_dim = data['X_seq'].shape[1]  # 序列特征维度
    struct_input_dim = data['X_struct'].shape[1]  # 结构特征维度
    
    model = MultiModalEpitopePredictor(
        seq_input_dim=seq_input_dim,
        struct_input_dim=struct_input_dim,
        hidden_dims=[64, 32],
        num_classes=2,
        dropout=0.3
    )
    
    # 模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🧠 模型架构信息:")
    print(f"   📊 序列特征维度: {seq_input_dim}")
    print(f"   🏗️  结构特征维度: {struct_input_dim}")
    print(f"   🔢 总参数数量: {total_params:,}")
    print(f"   🎯 可训练参数: {trainable_params:,}")
    
    # 保存模型架构信息
    model_info = {
        'architecture': 'MultiModalEpitopePredictor',
        'seq_input_dim': seq_input_dim,
        'struct_input_dim': struct_input_dim,
        'hidden_dims': [64, 32],
        'num_classes': 2,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'features': {
            'sequence_features': ['length', 'hydrophobicity', 'net_charge'],
            'structure_features': [
                'sasa_mean', 'sasa_std', 'sasa_max',
                'curvature_mean', 'curvature_std',
                'depth_mean', 'depth_std',
                'electrostatic_mean', 'electrostatic_std',
                'hydrophobic_mean', 'hydrophobic_std'
            ]
        }
    }
    
    # 保存模型信息
    output_dir = Path("vaccine_pred/models")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "model_architecture_info.json", 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 模型架构设计完成!")
    print(f"💾 模型信息保存至: {output_dir}/model_architecture_info.json")
    print(f"\n🎯 下一步建议:")
    print(f"   1. 开始模型训练")
    print(f"   2. 调优超参数")
    print(f"   3. 评估模型性能")
    print(f"   4. 实现疫苗候选表位预测")
    
    return model, data_splits


if __name__ == "__main__":
    model, data_splits = main() 