#!/usr/bin/env python3
"""
结核病疫苗设计管道 - 命令行版本
用法：python 8vaccine_design_pipeline.py --input 输入.fasta --output 结果.csv [--max_epitopes 10]
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
import sys
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import itertools
import warnings

warnings.filterwarnings('ignore')

# 导入模型架构（请确保该文件在同一目录下）
from multimodal_model_architecture import MultiModalEpitopePredictor, EpitopeDataLoader


class VaccineDesignPipeline:
    """疫苗设计管道"""

    def __init__(self, model_path="best_multimodal_epitope_model.pth"):  # 直接当前目录
        self.model_path = Path(model_path)
        self.model = None
        self.data_loader = None
        self.scaler_seq = None
        self.scaler_struct = None

        # 疫苗设计参数
        self.epitope_length_range = (8, 25)
        self.similarity_threshold = 0.8
        self.min_immunogenicity_score = 0.5
        self.bcell_forced_ratio = 0.3
        self.bcell_threshold = 0.8
        self.top_n_candidates = 300

        self.linkers = {
            'flexible': 'GGGGS',
            'rigid': 'EAAAK',
            'cleavable': 'GPGPG',
            'immunogenic': 'KK'
        }

        print("🧬 疫苗设计管道初始化", file=sys.stderr)

    def load_trained_model(self):
        """加载训练好的模型"""
        try:
            print("📥 加载训练好的模型...", file=sys.stderr)

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

            print(f"✅ 模型加载成功!", file=sys.stderr)

            self.data_loader = EpitopeDataLoader()
            data = self.data_loader.load_and_preprocess_data()

            self.scaler_seq = self.data_loader.scaler_seq
            self.scaler_struct = self.data_loader.scaler_struct

            return True

        except Exception as e:
            print(f"❌ 模型加载失败: {e}", file=sys.stderr)
            return False

    def generate_candidate_epitopes(self, protein_sequence, epitope_lengths=[9, 11, 13, 15, 20, 25], step=2):
        """生成候选表位（步长2）"""
        candidates = []
        seq_len = len(protein_sequence)

        for length in epitope_lengths:
            if length < seq_len:
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

        print(f"🧬 生成候选表位 (步长={step}): {len(candidates)}个", file=sys.stderr)
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
        """预测表位免疫原性"""
        if self.model is None:
            raise ValueError("请先加载训练好的模型")

        print(f"🔬 预测{len(epitope_candidates)}个候选表位的免疫原性...", file=sys.stderr)

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

        predictions.sort(key=lambda x: x['immunogenicity_score'], reverse=True)

        print(f"✅ 预测完成!", file=sys.stderr)
        return predictions

    def filter_top_candidates(self, predictions, top_n=None):
        if top_n is None:
            top_n = self.top_n_candidates
        if len(predictions) <= top_n:
            return predictions
        filtered = predictions[:top_n]
        print(f"📊 保留免疫原性前 {top_n} 个候选", file=sys.stderr)
        return filtered

    def _calculate_sequence_features(self, sequence):
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
        print(f"🔄 去除相似度>{similarity_threshold}的表位...", file=sys.stderr)
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

        print(f"✅ 去重完成: 保留{len(unique_epitopes)}个", file=sys.stderr)
        return unique_epitopes

    def _calculate_sequence_similarity(self, seq1, seq2):
        if len(seq1) != len(seq2):
            lcs_length = self._longest_common_subsequence(seq1, seq2)
            return lcs_length / max(len(seq1), len(seq2))
        else:
            matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
            return matches / len(seq1)

    def _longest_common_subsequence(self, seq1, seq2):
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
        print(f"🎯 选择最优表位组合 (最多{max_epitopes}个)...", file=sys.stderr)

        high_quality = [
            p for p in predictions
            if p['immunogenicity_score'] >= self.min_immunogenicity_score
        ]

        if len(high_quality) == 0:
            print("⚠️  没有找到高质量表位，降低阈值...", file=sys.stderr)
            high_quality = predictions[:max_epitopes * 2]

        if balance_bcell_tcell:
            bcell_epitopes = [p for p in high_quality if p['predicted_type'] == 'B-cell']
            tcell_epitopes = [p for p in high_quality if p['predicted_type'] == 'T-cell']

            target_bcell = max(1, int(max_epitopes * self.bcell_forced_ratio))
            target_tcell = max_epitopes - target_bcell

            if len(bcell_epitopes) < target_bcell:
                needed = target_bcell - len(bcell_epitopes)
                sorted_by_length = sorted(
                    [p for p in high_quality if p not in tcell_epitopes and p not in bcell_epitopes],
                    key=lambda x: len(x['sequence']), reverse=True
                )
                forced_bcell = sorted_by_length[:needed]
                for p in forced_bcell:
                    p['predicted_type'] = 'B-cell'
                    bcell_epitopes.append(p)

            if len(tcell_epitopes) < target_tcell:
                needed = target_tcell - len(tcell_epitopes)
                remaining = [p for p in high_quality if p not in bcell_epitopes and p not in tcell_epitopes]
                tcell_epitopes.extend(remaining[:needed])

            selected = bcell_epitopes[:target_bcell] + tcell_epitopes[:target_tcell]
        else:
            selected = high_quality[:max_epitopes]

        selected.sort(key=lambda x: x['immunogenicity_score'], reverse=True)

        print(f"✅ 选择完成: {len(selected)}个表位", file=sys.stderr)
        return selected

    def design_multitope_vaccine(self, selected_epitopes, linker_type='flexible'):
        """设计多表位疫苗（返回疫苗序列和表位详情）"""
        print(f"🧬 设计多表位疫苗...", file=sys.stderr)

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

        print(f"✅ 疫苗设计完成!", file=sys.stderr)
        return vaccine_design

    def run_vaccine_design_pipeline(self, protein_sequence, max_epitopes=10, output_csv=None):
        print("🚀 启动结核病疫苗设计流程", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        if not self.load_trained_model():
            return None

        candidates = self.generate_candidate_epitopes(protein_sequence)
        predictions = self.predict_epitope_immunogenicity(candidates)
        filtered = self.filter_top_candidates(predictions, self.top_n_candidates)
        unique = self.remove_similar_epitopes(filtered, self.similarity_threshold)
        selected = self.select_optimal_epitopes(unique, max_epitopes)
        vaccine_design = self.design_multitope_vaccine(selected)

        if output_csv:
            # 保存表位详情为CSV
            epitope_df = pd.DataFrame(vaccine_design['epitope_details'])
            epitope_df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"💾 表位详情已保存至: {output_csv}", file=sys.stderr)

            # 保存疫苗设计摘要为JSON（不包含epitope_details）
            output_path = Path(output_csv)
            summary_path = output_path.parent / (output_path.stem + "_vaccine_summary.json")
            summary = {k: v for k, v in vaccine_design.items() if k != 'epitope_details'}
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
            print(f"💾 疫苗设计摘要已保存至: {summary_path}", file=sys.stderr)

        return vaccine_design


def main():
    parser = argparse.ArgumentParser(description='结核杆菌表位预测和疫苗设计')
    parser.add_argument('--input', type=str, required=True, help='输入的FASTA文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出CSV文件路径')
    parser.add_argument('--max_epitopes', type=int, default=10, help='最终选择的表位数量')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误：输入文件 {input_path} 不存在", file=sys.stderr)
        sys.exit(1)

    try:
        with open(input_path, 'r') as f:
            records = list(SeqIO.parse(f, 'fasta'))
        if not records:
            print("错误：FASTA文件中没有序列", file=sys.stderr)
            sys.exit(1)
        protein_sequence = str(records[0].seq)
        print(f"从FASTA文件读取序列: {records[0].id}, 长度 {len(protein_sequence)}", file=sys.stderr)
    except Exception as e:
        print(f"读取FASTA文件失败: {e}", file=sys.stderr)
        sys.exit(1)

    pipeline = VaccineDesignPipeline()
    result = pipeline.run_vaccine_design_pipeline(
        protein_sequence=protein_sequence,
        max_epitopes=args.max_epitopes,
        output_csv=args.output
    )

    if result is None:
        print("疫苗设计失败", file=sys.stderr)
        sys.exit(1)

    print("🎉 处理完成！", file=sys.stderr)


if __name__ == "__main__":
    main()
