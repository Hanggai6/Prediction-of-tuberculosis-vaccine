#!/usr/bin/env python3
"""
结核病疫苗设计管道
================

功能：
1. 从结核杆菌蛋白质生成候选表位（步长2，减少数量）
2. 使用训练模型预测免疫原性
3. 按免疫原性排序，只保留top N候选
4. 去除同源性高的表位
5. 基于多种标准筛选最佳表位（强制引入B细胞）
6. 设计多表位疫苗构建体
7. 生成疫苗候选序列

"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import json
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import itertools
import warnings

warnings.filterwarnings('ignore')

# 导入模型架构
from multimodal_model_architecture import MultiModalEpitopePredictor, EpitopeDataLoader


class VaccineDesignPipeline:
    """疫苗设计管道"""

    def __init__(self, model_path="models/best_multimodal_epitope_model.pth"):
        self.model_path = Path(model_path)
        self.model = None
        self.data_loader = None
        self.scaler_seq = None
        self.scaler_struct = None

        # 疫苗设计参数
        self.epitope_length_range = (8, 25)  # 表位长度范围
        self.similarity_threshold = 0.8  # 同源性阈值
        self.min_immunogenicity_score = 0.5  # 最小免疫原性评分（降低以便更多候选）

        # B细胞强制引入参数
        self.bcell_forced_ratio = 0.3  # 强制B细胞占比（约1/3）
        self.bcell_threshold = 0.8  # T细胞判定阈值（>0.8判为T细胞，否则判为B细胞）

        # 候选表位数控制
        self.top_n_candidates = 300  # 预测后只保留前N个高免疫原性表位

        # 常用linker序列
        self.linkers = {
            'flexible': 'GGGGS',  # 柔性连接子
            'rigid': 'EAAAK',  # 刚性连接子
            'cleavable': 'GPGPG',  # 可切割连接子
            'immunogenic': 'KK'  # 免疫增强连接子
        }

        print("🧬 疫苗设计管道初始化")

    def load_trained_model(self):
        """加载训练好的模型"""
        try:
            print("📥 加载训练好的模型...")

            if not self.model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(self.model_path, map_location=device)

            self.model = MultiModalEpitopePredictor(
                seq_input_dim=3,
                struct_input_dim=11,
                hidden_dims=[64, 32],
                num_classes=2,
                dropout=0.3
            )

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(device)
            self.model.eval()

            print(f"✅ 模型加载成功!")
            print(f"   📊 最佳Epoch: {checkpoint.get('epoch', -1) + 1}")
            print(f"   🎯 验证准确率: {checkpoint.get('val_accuracy_main', 0):.4f}")

            self.data_loader = EpitopeDataLoader()
            data = self.data_loader.load_and_preprocess_data()

            self.scaler_seq = self.data_loader.scaler_seq
            self.scaler_struct = self.data_loader.scaler_struct

            return True

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False

    def generate_candidate_epitopes(self, protein_sequence, epitope_lengths=[9, 11, 13, 15, 20, 25], step=2):
        """
        从蛋白质序列生成候选表位
        使用步长 step 减少窗口数量，避免生成过多候选
        """
        candidates = []
        seq_len = len(protein_sequence)

        for length in epitope_lengths:
            if length < seq_len:
                # 步长为 step，只取起始位置为 step 倍数的窗口
                for i in range(0, seq_len - length + 1, step):
                    epitope = protein_sequence[i:i + length]

                    if self._is_valid_epitope(epitope):
                        candidates.append({
                            'sequence': epitope,
                            'start_pos': i + 1,
                            'end_pos': i + length,
                            'length': length,
                            'source_protein': 'input_protein'
                        })

        print(f"🧬 生成候选表位 (步长={step}): {len(candidates)}个")
        return candidates

    def _is_valid_epitope(self, sequence):
        """检查表位序列是否有效"""
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        if not all(aa in valid_aa for aa in sequence.upper()):
            return False
        if len(set(sequence)) < 3:
            return False
        for i in range(len(sequence) - 2):
            if sequence[i] == sequence[i + 1] == sequence[i + 2]:
                return False
        return True

    def predict_epitope_immunogenicity(self, epitope_candidates):
        """预测表位免疫原性（调整B细胞判定阈值）"""
        if self.model is None:
            raise ValueError("请先加载训练好的模型")

        print(f"🔬 预测{len(epitope_candidates)}个候选表位的免疫原性...")

        predictions = []
        device = next(self.model.parameters()).device

        for candidate in epitope_candidates:
            sequence = candidate['sequence']

            seq_features = self._calculate_sequence_features(sequence)
            struct_features = np.zeros(11)
            has_structure = False

            seq_features_scaled = self.scaler_seq.transform([seq_features])[0]
            struct_features_scaled = self.scaler_struct.transform([struct_features])[0]

            seq_tensor = torch.FloatTensor(seq_features_scaled).unsqueeze(0).to(device)
            struct_tensor = torch.FloatTensor(struct_features_scaled).unsqueeze(0).to(device)
            has_struct_tensor = torch.BoolTensor([has_structure]).to(device)

            with torch.no_grad():
                outputs = self.model(seq_tensor, struct_tensor, has_struct_tensor)

                immunogenicity_prob = torch.softmax(outputs['main_prediction'], dim=1)[0, 1].item()
                epitope_type_prob = torch.softmax(outputs['epitope_type_prediction'], dim=1)[0, 1].item()

                # 使用更高的阈值判定T细胞，否则判为B细胞
                predicted_type = 'T-cell' if epitope_type_prob > self.bcell_threshold else 'B-cell'

                prediction = candidate.copy()
                prediction.update({
                    'immunogenicity_score': immunogenicity_prob,
                    'tcell_probability': epitope_type_prob,
                    'bcell_probability': 1 - epitope_type_prob,
                    'predicted_immunogenic': immunogenicity_prob > 0.5,
                    'predicted_type': predicted_type
                })

                predictions.append(prediction)

        # 按免疫原性评分排序
        predictions.sort(key=lambda x: x['immunogenicity_score'], reverse=True)

        print(f"✅ 预测完成!")
        print(f"   🎯 高免疫原性表位 (>0.7): {len([p for p in predictions if p['immunogenicity_score'] > 0.7])}")
        print(f"   🎯 T细胞表位: {len([p for p in predictions if p['predicted_type'] == 'T-cell'])}")
        print(f"   🎯 B细胞表位: {len([p for p in predictions if p['predicted_type'] == 'B-cell'])}")

        return predictions

    def filter_top_candidates(self, predictions, top_n=None):
        """只保留免疫原性最高的前 top_n 个候选"""
        if top_n is None:
            top_n = self.top_n_candidates
        if len(predictions) <= top_n:
            return predictions
        filtered = predictions[:top_n]
        print(f"📊 保留免疫原性前 {top_n} 个候选 (共 {len(predictions)} 个)")
        return filtered

    def _calculate_sequence_features(self, sequence):
        """计算表位序列特征"""
        aa_properties = {
            'A': {'hydrophobic': 1.8, 'volume': 88.6, 'charge': 0},
            'R': {'hydrophobic': -4.5, 'volume': 173.4, 'charge': 1},
            'N': {'hydrophobic': -3.5, 'volume': 114.1, 'charge': 0},
            'D': {'hydrophobic': -3.5, 'volume': 111.1, 'charge': -1},
            'C': {'hydrophobic': 2.5, 'volume': 108.5, 'charge': 0},
            'Q': {'hydrophobic': -3.5, 'volume': 143.8, 'charge': 0},
            'E': {'hydrophobic': -3.5, 'volume': 138.4, 'charge': -1},
            'G': {'hydrophobic': -0.4, 'volume': 60.1, 'charge': 0},
            'H': {'hydrophobic': -3.2, 'volume': 153.2, 'charge': 0.5},
            'I': {'hydrophobic': 4.5, 'volume': 166.7, 'charge': 0},
            'L': {'hydrophobic': 3.8, 'volume': 166.7, 'charge': 0},
            'K': {'hydrophobic': -3.9, 'volume': 168.6, 'charge': 1},
            'M': {'hydrophobic': 1.9, 'volume': 162.9, 'charge': 0},
            'F': {'hydrophobic': 2.8, 'volume': 189.9, 'charge': 0},
            'P': {'hydrophobic': -1.6, 'volume': 112.7, 'charge': 0},
            'S': {'hydrophobic': -0.8, 'volume': 89.0, 'charge': 0},
            'T': {'hydrophobic': -0.7, 'volume': 116.1, 'charge': 0},
            'W': {'hydrophobic': -0.9, 'volume': 227.8, 'charge': 0},
            'Y': {'hydrophobic': -1.3, 'volume': 193.6, 'charge': 0},
            'V': {'hydrophobic': 4.2, 'volume': 140.0, 'charge': 0}
        }

        length = len(sequence)
        hydrophobicity = 0
        net_charge = 0

        for aa in sequence.upper():
            if aa in aa_properties:
                hydrophobicity += aa_properties[aa]['hydrophobic']
                net_charge += aa_properties[aa]['charge']

        if length > 0:
            hydrophobicity /= length
            net_charge /= length

        return [length, hydrophobicity, net_charge]

    def remove_similar_epitopes(self, predictions, similarity_threshold=0.8):
        """去除相似的表位"""
        print(f"🔄 去除相似度>{similarity_threshold}的表位...")

        unique_epitopes = []
        removed_count = 0

        for pred in predictions:
            is_similar = False
            for unique_pred in unique_epitopes:
                similarity = self._calculate_sequence_similarity(
                    pred['sequence'], unique_pred['sequence']
                )
                if similarity > similarity_threshold:
                    is_similar = True
                    removed_count += 1
                    break
            if not is_similar:
                unique_epitopes.append(pred)

        print(f"✅ 去重完成: 保留{len(unique_epitopes)}个, 移除{removed_count}个相似表位")
        return unique_epitopes

    def _calculate_sequence_similarity(self, seq1, seq2):
        """计算序列相似度"""
        if len(seq1) != len(seq2):
            lcs_length = self._longest_common_subsequence(seq1, seq2)
            return lcs_length / max(len(seq1), len(seq2))
        else:
            matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
            return matches / len(seq1)

    def _longest_common_subsequence(self, seq1, seq2):
        """计算最长公共子序列长度"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    def select_optimal_epitopes(self, predictions, max_epitopes=10, balance_bcell_tcell=True):
        """选择最优表位组合（强制引入B细胞）"""
        print(f"🎯 选择最优表位组合 (最多{max_epitopes}个)...")

        # 筛选高质量表位
        high_quality = [
            p for p in predictions
            if p['immunogenicity_score'] >= self.min_immunogenicity_score
        ]

        if len(high_quality) == 0:
            print("⚠️  没有找到高质量表位，降低阈值...")
            high_quality = predictions[:max_epitopes * 2]

        if balance_bcell_tcell:
            # 按预测类型分离
            bcell_epitopes = [p for p in high_quality if p['predicted_type'] == 'B-cell']
            tcell_epitopes = [p for p in high_quality if p['predicted_type'] == 'T-cell']

            # 目标比例：1:2 (B细胞:T细胞) 或根据参数
            target_bcell = max(1, int(max_epitopes * self.bcell_forced_ratio))
            target_tcell = max_epitopes - target_bcell

            # 如果B细胞不足，从长肽段中强制补充
            if len(bcell_epitopes) < target_bcell:
                needed = target_bcell - len(bcell_epitopes)
                print(f"⚠️  B细胞表位不足，将从长肽段中强制选择 {needed} 个作为B细胞")

                # 按长度排序，取最长的且不在tcell_epitopes中的
                sorted_by_length = sorted(
                    [p for p in high_quality if p not in tcell_epitopes and p not in bcell_epitopes],
                    key=lambda x: len(x['sequence']), reverse=True
                )
                forced_bcell = sorted_by_length[:needed]
                for p in forced_bcell:
                    p['predicted_type'] = 'B-cell'  # 强制标记为B细胞
                    bcell_epitopes.append(p)

            # 如果T细胞不足，从剩余中补充（保持类型）
            if len(tcell_epitopes) < target_tcell:
                needed = target_tcell - len(tcell_epitopes)
                remaining = [p for p in high_quality if p not in bcell_epitopes and p not in tcell_epitopes]
                tcell_epitopes.extend(remaining[:needed])

            # 最终选择
            selected = bcell_epitopes[:target_bcell] + tcell_epitopes[:target_tcell]
        else:
            selected = high_quality[:max_epitopes]

        # 按免疫原性排序
        selected.sort(key=lambda x: x['immunogenicity_score'], reverse=True)

        print(f"✅ 选择完成:")
        print(f"   🎯 总选择: {len(selected)}个表位")
        print(f"   🎯 B细胞表位: {len([p for p in selected if p['predicted_type'] == 'B-cell'])}个")
        print(f"   🎯 T细胞表位: {len([p for p in selected if p['predicted_type'] == 'T-cell'])}个")
        print(f"   🎯 平均免疫原性: {np.mean([p['immunogenicity_score'] for p in selected]):.3f}")

        return selected

    def design_multitope_vaccine(self, selected_epitopes, linker_type='flexible'):
        """设计多表位疫苗"""
        print(f"🧬 设计多表位疫苗...")

        if not selected_epitopes:
            raise ValueError("没有选择的表位用于疫苗设计")

        linker = self.linkers.get(linker_type, self.linkers['flexible'])

        epitope_sequences = [ep['sequence'] for ep in selected_epitopes]
        vaccine_sequence = linker.join(epitope_sequences)

        vaccine_analysis = ProteinAnalysis(vaccine_sequence)

        vaccine_design = {
            'vaccine_sequence': vaccine_sequence,
            'total_length': len(vaccine_sequence),
            'num_epitopes': len(selected_epitopes),
            'linker_type': linker_type,
            'linker_sequence': linker,
            'molecular_weight': vaccine_analysis.molecular_weight(),
            'isoelectric_point': vaccine_analysis.isoelectric_point(),
            'epitope_composition': {
                'bcell_epitopes': len([ep for ep in selected_epitopes if ep['predicted_type'] == 'B-cell']),
                'tcell_epitopes': len([ep for ep in selected_epitopes if ep['predicted_type'] == 'T-cell'])
            },
            'average_immunogenicity': np.mean([ep['immunogenicity_score'] for ep in selected_epitopes]),
            'epitope_details': selected_epitopes
        }

        print(f"✅ 疫苗设计完成:")
        print(f"   🧬 疫苗长度: {vaccine_design['total_length']}个氨基酸")
        print(f"   🎯 包含表位: {vaccine_design['num_epitopes']}个")
        print(f"   ⚖️  分子量: {vaccine_design['molecular_weight']:.1f} Da")
        print(f"   📊 等电点: {vaccine_design['isoelectric_point']:.2f}")
        print(f"   🎯 平均免疫原性: {vaccine_design['average_immunogenicity']:.3f}")

        return vaccine_design

    def save_results(self, vaccine_design, output_dir="results/vaccine_design"):
        """保存疫苗设计结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result_file = output_dir / "vaccine_design_results.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(vaccine_design, f, ensure_ascii=False, indent=2, default=str)

        fasta_file = output_dir / "vaccine_sequence.fasta"
        with open(fasta_file, 'w') as f:
            f.write(">Tuberculosis_Multitope_Vaccine\n")
            f.write(vaccine_design['vaccine_sequence'] + "\n")

        epitope_df = pd.DataFrame(vaccine_design['epitope_details'])
        csv_file = output_dir / "selected_epitopes.csv"
        epitope_df.to_csv(csv_file, index=False, encoding='utf-8')

        print(f"💾 结果保存至:")
        print(f"   📄 详细结果: {result_file}")
        print(f"   🧬 疫苗序列: {fasta_file}")
        print(f"   📊 表位详情: {csv_file}")

        return output_dir

    def run_vaccine_design_pipeline(self, protein_sequence, max_epitopes=10):
        """运行完整的疫苗设计流程"""
        print("🚀 启动结核病疫苗设计流程")
        print("=" * 60)

        if not self.load_trained_model():
            return None

        # 生成候选表位（使用步长2，减少数量）
        candidates = self.generate_candidate_epitopes(
            protein_sequence,
            epitope_lengths=[9, 11, 13, 15, 20, 25],
            step=2
        )

        # 预测免疫原性
        predictions = self.predict_epitope_immunogenicity(candidates)

        # 只保留免疫原性最高的 top N 个候选
        filtered_predictions = self.filter_top_candidates(predictions, self.top_n_candidates)

        # 去除相似表位
        unique_predictions = self.remove_similar_epitopes(filtered_predictions, self.similarity_threshold)

        # 选择最优表位
        selected_epitopes = self.select_optimal_epitopes(unique_predictions, max_epitopes)

        # 设计多表位疫苗
        vaccine_design = self.design_multitope_vaccine(selected_epitopes)

        # 保存结果
        output_dir = self.save_results(vaccine_design)

        print(f"\n🎉 疫苗设计完成!")
        print(f"📋 疫苗摘要:")
        print(f"   🧬 疫苗序列: {vaccine_design['vaccine_sequence'][:50]}...")
        print(f"   📏 总长度: {vaccine_design['total_length']}氨基酸")
        print(f"   🎯 表位数量: {vaccine_design['num_epitopes']}个")
        print(f"   📊 平均免疫原性: {vaccine_design['average_immunogenicity']:.3f}")

        return vaccine_design


def main():
    """演示疫苗设计流程"""
    # 使用 Ag85B 蛋白序列（B细胞和T细胞表位丰富）
    ag85b_sequence = "FSRPGLPVEYLQVPSPSMGRDIKVQFQSGGANSPALYLLDGLRAQDDFSGWDINTPAFEWYDQSGLSVVMPVGGQSSFYSDWYQPACGKAGCQTYKWETFLTSELPGWLQANRHVKPTGSAVVGLSMAASSALTLAIYHPQQFVYAGAMSGLLDPSQAMGPTLIGLAMGDAGGYKASDMWGPKEDPAWQRNDPLLNVGKLIANNTRVWVYCGNGKPSDLGGNNLPAKFLEGFVRTSNIKFQDAYNAGGGHNGVFDFPDSGTHSWEYWGAQLNAMKPDLQRALGATPNTGPAPQGA"

    vaccine_designer = VaccineDesignPipeline()

    # 运行设计流程，允许最多8个表位
    result = vaccine_designer.run_vaccine_design_pipeline(
        protein_sequence=ag85b_sequence,
        max_epitopes=8
    )

    return result


if __name__ == "__main__":
    vaccine_result = main()