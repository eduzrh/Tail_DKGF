import os
from collections import defaultdict
import argparse

def load_node_degrees(degree_file):
    """加载节点度字典"""
    degree_dict = {}
    with open(degree_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                node_id, degree = parts
                degree_dict[node_id] = int(degree)
    return degree_dict

def extract_triples_by_degree(test_file, degree_dict, target_degrees=None, is_long_tail=True):
    """
    从测试文件中提取指定度的三元组
    
    Args:
        test_file: 测试文件路径
        degree_dict: 节点度字典
        target_degrees: 目标度列表，如 [1, 2, 3, 4] 或 None（表示所有长尾或非长尾）
        is_long_tail: True表示提取长尾(度<=4), False表示提取非长尾(度>=5)
    
    Returns:
        dict: {degree: [triples]}, 以及总数统计
    """
    triples_by_degree = defaultdict(list)
    total_count = 0
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_count += 1
            parts = line.split('\t')
            if len(parts) >= 3:
                head, tail = parts[0], parts[2]
                head_deg = degree_dict.get(head, 0)  # 未知节点度为0
                tail_deg = degree_dict.get(tail, 0)
                
                # 取头尾实体中度较小的作为该三元组的"长尾度"
                min_deg = min(head_deg, tail_deg)
                
                if is_long_tail:
                    # 长尾节点: 度 <= 4
                    if min_deg <= 4:
                        if target_degrees is None or min_deg in target_degrees:
                            triples_by_degree[min_deg].append(line)
                else:
                    # 非长尾节点: 度 >= 5
                    if min_deg >= 5:
                        triples_by_degree['non_long_tail'].append(line)
    
    return dict(triples_by_degree), total_count

def process_dataset(dataset_path, output_dir, extract_by_degree=True):
    """
    处理单个数据集
    
    Args:
        dataset_path: 数据集路径
        output_dir: 输出目录
        extract_by_degree: True=按度分别保存, False=长尾和非长尾各保存一个文件
    """
    dataset_name = os.path.basename(dataset_path)
    print(f"\n{'='*60}")
    print(f"处理数据集: {dataset_name}")
    print(f"{'='*60}")
    
    # 1. 加载节点度
    degree_file = os.path.join(dataset_path, 'node_degrees.txt')
    if not os.path.exists(degree_file):
        print(f"错误: 节点度文件不存在: {degree_file}")
        print("请先运行: python compute_node_degrees.py")
        return None
    
    degree_dict = load_node_degrees(degree_file)
    print(f"加载节点度: {len(degree_dict)} 个节点")
    
    # 统计度分布
    degree_counts = defaultdict(int)
    for deg in degree_dict.values():
        if deg <= 4:
            degree_counts[deg] += 1
        else:
            degree_counts['≥5'] += 1
    
    print(f"\n节点度分布:")
    for deg in range(0, 5):
        print(f"  度={deg}: {degree_counts.get(deg, 0)} 节点")
    print(f"  度≥5: {degree_counts.get('≥5', 0)} 节点")
    
    # 2. 获取测试文件路径
    gold_file = os.path.join(dataset_path, 'ref_triples_1_fusion')
    false_file = os.path.join(dataset_path, 'false_ref_triples_1_fusion')
    
    if not os.path.exists(gold_file):
        print(f"错误: 正样本文件不存在: {gold_file}")
        return None
    if not os.path.exists(false_file):
        print(f"错误: 负样本文件不存在: {false_file}")
        return None
    
    # 3. 创建输出目录
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    stats = {
        'dataset': dataset_name,
        'pos_stats': {},
        'neg_stats': {}
    }
    
    # 4. 处理正样本
    print(f"\n{'='*50}")
    print("提取正样本...")
    print(f"{'='*50}")
    
    # 长尾节点 (度 0-4)
    pos_long_tail, total_pos = extract_triples_by_degree(
        gold_file, degree_dict, target_degrees=None, is_long_tail=True
    )
    
    # 非长尾节点 (度 >=5)
    pos_non_long_tail, _ = extract_triples_by_degree(
        gold_file, degree_dict, target_degrees=None, is_long_tail=False
    )
    
    print(f"原始正样本总数: {total_pos}")
    print(f"\n长尾节点正样本分布:")
    total_long_tail_pos = 0
    for deg in range(0, 5):
        count = len(pos_long_tail.get(deg, []))
        total_long_tail_pos += count
        print(f"  度={deg}: {count} ({count/total_pos*100:.2f}%)")
    
    non_long_tail_pos = len(pos_non_long_tail.get('non_long_tail', []))
    print(f"\n非长尾节点正样本:")
    print(f"  度≥5: {non_long_tail_pos} ({non_long_tail_pos/total_pos*100:.2f}%)")
    
    stats['pos_stats']['total'] = total_pos
    stats['pos_stats']['long_tail'] = total_long_tail_pos
    stats['pos_stats']['non_long_tail'] = non_long_tail_pos
    
    # 5. 处理负样本
    print(f"\n{'='*50}")
    print("提取负样本...")
    print(f"{'='*50}")
    
    # 长尾节点 (度 0-4)
    neg_long_tail, total_neg = extract_triples_by_degree(
        false_file, degree_dict, target_degrees=None, is_long_tail=True
    )
    
    # 非长尾节点 (度 >=5)
    neg_non_long_tail, _ = extract_triples_by_degree(
        false_file, degree_dict, target_degrees=None, is_long_tail=False
    )
    
    print(f"原始负样本总数: {total_neg}")
    print(f"\n长尾节点负样本分布:")
    total_long_tail_neg = 0
    for deg in range(0, 5):
        count = len(neg_long_tail.get(deg, []))
        total_long_tail_neg += count
        print(f"  度={deg}: {count} ({count/total_neg*100:.2f}%)")
    
    non_long_tail_neg = len(neg_non_long_tail.get('non_long_tail', []))
    print(f"\n非长尾节点负样本:")
    print(f"  度≥5: {non_long_tail_neg} ({non_long_tail_neg/total_neg*100:.2f}%)")
    
    stats['neg_stats']['total'] = total_neg
    stats['neg_stats']['long_tail'] = total_long_tail_neg
    stats['neg_stats']['non_long_tail'] = non_long_tail_neg
    
    # 6. 保存文件
    print(f"\n{'='*50}")
    print("保存文件...")
    print(f"{'='*50}")
    
    if extract_by_degree:
        # 按度分别保存
        for deg in range(0, 5):
            # 正样本
            if deg in pos_long_tail:
                pos_file = os.path.join(dataset_output_dir, f'long_tail_deg_{deg}_pos.txt')
                with open(pos_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(pos_long_tail[deg]))
                print(f"  度={deg} 正样本: {pos_file} ({len(pos_long_tail[deg])} 条)")
            
            # 负样本
            if deg in neg_long_tail:
                neg_file = os.path.join(dataset_output_dir, f'long_tail_deg_{deg}_neg.txt')
                with open(neg_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(neg_long_tail[deg]))
                print(f"  度={deg} 负样本: {neg_file} ({len(neg_long_tail[deg])} 条)")
    
    # 保存合并的长尾数据
    all_long_tail_pos = []
    all_long_tail_neg = []
    for deg in range(0, 5):
        all_long_tail_pos.extend(pos_long_tail.get(deg, []))
        all_long_tail_neg.extend(neg_long_tail.get(deg, []))
    
    pos_merged_file = os.path.join(dataset_output_dir, 'long_tail_all_pos.txt')
    with open(pos_merged_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_long_tail_pos))
    print(f"\n  长尾合并 正样本: {pos_merged_file} ({len(all_long_tail_pos)} 条)")
    
    neg_merged_file = os.path.join(dataset_output_dir, 'long_tail_all_neg.txt')
    with open(neg_merged_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_long_tail_neg))
    print(f"  长尾合并 负样本: {neg_merged_file} ({len(all_long_tail_neg)} 条)")
    
    # 保存非长尾数据
    non_long_tail_pos_file = os.path.join(dataset_output_dir, 'non_long_tail_pos.txt')
    with open(non_long_tail_pos_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(pos_non_long_tail.get('non_long_tail', [])))
    print(f"\n  非长尾 正样本: {non_long_tail_pos_file} ({non_long_tail_pos} 条)")
    
    non_long_tail_neg_file = os.path.join(dataset_output_dir, 'non_long_tail_neg.txt')
    with open(non_long_tail_neg_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(neg_non_long_tail.get('non_long_tail', [])))
    print(f"  非长尾 负样本: {non_long_tail_neg_file} ({non_long_tail_neg} 条)")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='从测试集中提取长尾和非长尾数据')
    parser.add_argument('--base_dir', default='./data', help='Root directory containing the dataset folders')
    parser.add_argument('--output_dir', default='./outputs/long_tail_data', help='Output directory')
    parser.add_argument('--datasets', nargs='+', default=['W-I-S1', 'W-I-S2', 'Y-I-S1', 'Y-I-S2'], help='数据集列表')
    parser.add_argument('--extract_by_degree', action='store_true', help='是否按度分别保存文件')
    
    args = parser.parse_args()
    
    print("="*60)
    print("长尾与非长尾测试数据提取工具")
    print("="*60)
    print(f"数据集根目录: {args.base_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"长尾定义: 度 ∈ [0, 1, 2, 3, 4]")
    print(f"非长尾定义: 度 >= 5")
    print(f"按度分别保存: {'是' if args.extract_by_degree else '否'}")
    print(f"数据集列表: {', '.join(args.datasets)}")
    
    all_stats = []
    for dataset in args.datasets:
        dataset_path = os.path.join(args.base_dir, dataset)
        if os.path.exists(dataset_path):
            stats = process_dataset(dataset_path, args.output_dir, args.extract_by_degree)
            if stats:
                all_stats.append(stats)
        else:
            print(f"\n警告: 数据集不存在 {dataset_path}")
    
    # 输出总体摘要
    print("\n" + "="*80)
    print("总体摘要")
    print("="*80)
    print(f"{'数据集':<12} {'总正样本':<10} {'长尾正':<10} {'非长尾正':<10} "
          f"{'总负样本':<10} {'长尾负':<10} {'非长尾负':<10}")
    print("-"*80)
    
    for s in all_stats:
        print(f"{s['dataset']:<12} "
              f"{s['pos_stats']['total']:<10} "
              f"{s['pos_stats']['long_tail']:<10} "
              f"{s['pos_stats']['non_long_tail']:<10} "
              f"{s['neg_stats']['total']:<10} "
              f"{s['neg_stats']['long_tail']:<10} "
              f"{s['neg_stats']['non_long_tail']:<10}")
    
    print("\n文件命名规则:")
    print("  long_tail_deg_X_pos.txt : 度=X的长尾正样本 (X ∈ [0,1,2,3,4])")
    print("  long_tail_deg_X_neg.txt : 度=X的长尾负样本")
    print("  long_tail_all_pos.txt   : 所有长尾正样本合并")
    print("  long_tail_all_neg.txt   : 所有长尾负样本合并")
    print("  non_long_tail_pos.txt   : 非长尾正样本 (度≥5)")
    print("  non_long_tail_neg.txt   : 非长尾负样本 (度≥5)")

if __name__ == "__main__":
    main()