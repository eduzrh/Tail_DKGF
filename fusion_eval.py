# -*- coding: utf-8 -*-
from collections import defaultdict
import os
import argparse
from typing import Tuple


def calculate_entity_performance(output_triples_path: str,
                                 relative_ent_ids_path: str,
                                 ent_ids_2_path: str) -> Tuple[float, float, float, float]:
    """
    计算相关实体发现的 ACC、P、R、F1
    output_triples_path: 模型输出的融合三元组文件（预测结果）
    relative_ent_ids_path: 标准答案 sum_relative_ent_ids_2
    ent_ids_2_path: KG2 实体 id 文件
    """
    # 读取 KG2 实体 ID 集合
    with open(ent_ids_2_path, 'r', encoding='utf-8') as f:
        kg2_ent_ids = {line.strip().split('\t')[0] for line in f if line.strip()}

    # 读取输出三元组中的 KG2 相关实体（自动去重）
    predicted_entities = set()
    all_lines = []
    with open(output_triples_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            all_lines.append(s)
            parts = s.split('\t')
            if len(parts) >= 3:
                head, _, tail = parts[:3]
                if head in kg2_ent_ids:
                    predicted_entities.add(head)
                if tail in kg2_ent_ids:
                    predicted_entities.add(tail)

    # 统计重复情况
    unique_lines = set(all_lines)
    if len(all_lines) > len(unique_lines):
        print(f"[INFO] Found {len(all_lines) - len(unique_lines)} duplicate lines in prediction file")

    # 读取标准答案集合（正例）
    with open(relative_ent_ids_path, 'r', encoding='utf-8') as f:
        gold_entities = {line.strip() for line in f if line.strip()}

    # 负例集合：KG2中不属于gold_entities的部分
    negative_entities = kg2_ent_ids - gold_entities

    # 计算 TP、FP、TN、FN
    tp = len(predicted_entities & gold_entities)         # 预测为正，实际为正
    fp = len(predicted_entities & negative_entities)     # 预测为正，实际为负
    fn = len(gold_entities - predicted_entities)         # 预测为负，实际为正
    tn = len(negative_entities - predicted_entities)     # 预测为负，实际为负

    # 计算 ACC、P、R、F1
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("Entity Discovery Performance:")
    print(f"  ACC={accuracy:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    return accuracy, precision, recall, f1


def calculate_fusion_performance(output_triples_path: str,
                                 gold_triples_path: str,
                                 false_triples_path: str) -> Tuple[float, float, float, float]:
    """
    计算融合三元组的 ACC、P、R、F1
    output_triples_path: 模型输出的融合三元组文件（预测结果）
    gold_triples_path: 标准答案 sum_triples_1_fusion
    false_triples_path: 负例 false_ref_triples_1_fusion
    """
    # 读取预测结果并去重
    with open(output_triples_path, 'r', encoding='utf-8') as f:
        all_predicted = [line.strip() for line in f if line.strip()]
        predicted_triples = set(all_predicted)

    # 统计重复情况
    if len(all_predicted) > len(predicted_triples):
        print(f"[INFO] Removed {len(all_predicted) - len(predicted_triples)} duplicate predictions")

    # 读取正例（标准答案）
    with open(gold_triples_path, 'r', encoding='utf-8') as f:
        gold_triples = {line.strip() for line in f if line.strip()}

    # 读取负例
    with open(false_triples_path, 'r', encoding='utf-8') as f:
        false_triples = {line.strip() for line in f if line.strip()}

    # 计算 TP、FP、TN、FN
    tp = len(predicted_triples & gold_triples)      # 预测为正，实际为正
    fp = len(predicted_triples & false_triples)     # 预测为正，实际为负
    fn = len(gold_triples - predicted_triples)      # 预测为负，实际为正
    tn = len(false_triples - predicted_triples)     # 预测为负，实际为负
    
    # 计算 ACC、P、R、F1
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("Fusion Triple Performance:")
    print(f"  ACC={accuracy:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    return accuracy, precision, recall, f1


def build_report(dataset_dir: str,
                 pred_path: str,
                 entity_metrics: Tuple[float, float, float, float],
                 fusion_metrics: Tuple[float, float, float, float]) -> str:
    ea, ep, er, ef1 = entity_metrics
    fa, fp, fr, ff1 = fusion_metrics
    lines = []
    lines.append(f"Dataset Dir: {dataset_dir}")
    lines.append(f"Pred File  : {pred_path}")
    lines.append("")
    lines.append("== Entity Discovery Performance ==")
    lines.append(f"ACC={ea:.4f}, P={ep:.4f}, R={er:.4f}, F1={ef1:.4f}")
    lines.append("")
    lines.append("== Fusion Triple Performance ==")
    lines.append(f"ACC={fa:.4f}, P={fp:.4f}, R={fr:.4f}, F1={ff1:.4f}")
    lines.append("")
    return "\n".join(lines)


def assert_file_exists(path: str, desc: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{desc} 不存在：{path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fusion triples & entity discovery performance")
    parser.add_argument("--data_dir", type=str, default='./W-I-S1', required=True, help="数据集根目录（包含参考文件）")
    parser.add_argument("--pred", type=str, default='./outputs/W-I-S1/compare/pred_ours.txt', required=True, help="预测结果文件路径（每个方法各自的 pred_xxx.txt）")
    parser.add_argument("--out", type=str, default=None, help="评估结果输出文件路径（可选）")
    args = parser.parse_args()

    # 基准文件路径（基于 data_dir）
    sum_relative_ent_ids_2 = os.path.join(args.data_dir, "ref_relative_ent_ids_2")
    sum_triples_1_fusion = os.path.join(args.data_dir, "ref_triples_1_fusion")
    false_sum_triples_fusion = os.path.join(args.data_dir, "false_ref_triples_1_fusion")
    ent_ids_2 = os.path.join(args.data_dir, "ent_ids_2")

    # 基础校验
    assert_file_exists(args.pred, "预测结果文件")
    assert_file_exists(sum_relative_ent_ids_2, "标准答案（相关实体）")
    assert_file_exists(sum_triples_1_fusion, "标准答案（融合三元组-正例）")
    assert_file_exists(false_sum_triples_fusion, "标准答案（融合三元组-负例）")
    assert_file_exists(ent_ids_2, "KG2 实体 ID 文件")

    # 评估
    entity_metrics = calculate_entity_performance(
        args.pred, sum_relative_ent_ids_2, ent_ids_2
    )
    fusion_metrics = calculate_fusion_performance(
        args.pred, sum_triples_1_fusion, false_sum_triples_fusion
    )

    # 汇总输出
    report_text = build_report(args.data_dir, args.pred, entity_metrics, fusion_metrics)
    print("\n" + report_text)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report_text)
