import os
from collections import defaultdict

def load_all_entities(dataset_path):
    """加载数据集中的所有实体ID"""
    all_entities = set()
    
    ent_files = ['ent_ids_1', 'ent_ids_2']
    for ent_file in ent_files:
        ent_path = os.path.join(dataset_path, ent_file)
        if os.path.exists(ent_path):
            with open(ent_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        ent_id = parts[0]
                        all_entities.add(ent_id)
    
    return all_entities

def compute_node_degrees(triples_file_1, triples_file_2, all_entities=None):
    """
    计算所有节点的度(入度+出度)
    
    Args:
        triples_file_1: 第一个三元组文件路径
        triples_file_2: 第二个三元组文件路径
        all_entities: 所有实体ID集合（如果提供，会初始化所有节点度为0）
    
    Returns:
        dict: {node_id: degree}
    """
    # 初始化所有节点度为0
    if all_entities:
        degree_dict = {ent: 0 for ent in all_entities}
    else:
        degree_dict = defaultdict(int)
    
    # 读取两个三元组文件
    for triples_file in [triples_file_1, triples_file_2]:
        if not os.path.exists(triples_file):
            print(f"警告: 文件不存在 {triples_file}")
            continue
            
        with open(triples_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    head = parts[0]
                    tail = parts[2]
                    
                    # 头实体出度+1, 尾实体入度+1
                    if head in degree_dict:
                        degree_dict[head] += 1
                    else:
                        degree_dict[head] = 1
                    
                    if tail in degree_dict:
                        degree_dict[tail] += 1
                    else:
                        degree_dict[tail] = 1
    
    return dict(degree_dict)

def save_degree_dict(degree_dict, output_file):
    """保存节点度字典到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        # 按节点ID排序（尝试转为int排序，失败则按字符串排序）
        try:
            sorted_items = sorted(degree_dict.items(), key=lambda x: int(x[0]))
        except ValueError:
            sorted_items = sorted(degree_dict.items())
        
        for node_id, degree in sorted_items:
            f.write(f"{node_id}\t{degree}\n")
    
    print(f"节点度已保存到: {output_file}")
    print(f"总节点数: {len(degree_dict)}")
    
    # 统计不同度的节点分布
    degree_distribution = defaultdict(int)
    for deg in degree_dict.values():
        degree_distribution[deg] += 1
    
    print(f"\n节点度分布:")
    print(f"  度=0: {degree_distribution[0]} ({degree_distribution[0]/len(degree_dict)*100:.2f}%)")
    print(f"  度=1: {degree_distribution[1]} ({degree_distribution[1]/len(degree_dict)*100:.2f}%)")
    print(f"  度=2: {degree_distribution[2]} ({degree_distribution[2]/len(degree_dict)*100:.2f}%)")
    print(f"  度=3: {degree_distribution[3]} ({degree_distribution[3]/len(degree_dict)*100:.2f}%)")
    print(f"  度=4: {degree_distribution[4]} ({degree_distribution[4]/len(degree_dict)*100:.2f}%)")
    print(f"  度≥5: {sum(v for k, v in degree_distribution.items() if k >= 5)} "
          f"({sum(v for k, v in degree_distribution.items() if k >= 5)/len(degree_dict)*100:.2f}%)")
    
    # 统计长尾节点(度≤4)
    long_tail_nodes = [node for node, deg in degree_dict.items() if deg <= 4]
    print(f"\n长尾节点(度≤4): {len(long_tail_nodes)} ({len(long_tail_nodes)/len(degree_dict)*100:.2f}%)")

def process_dataset(dataset_path):
    """处理单个数据集"""
    triples_1 = os.path.join(dataset_path, 'triples_1')
    triples_2 = os.path.join(dataset_path, 'triples_2')
    
    print(f"\n{'='*60}")
    print(f"处理数据集: {os.path.basename(dataset_path)}")
    print(f"{'='*60}")
    
    # 加载所有实体
    print("加载所有实体ID...")
    all_entities = load_all_entities(dataset_path)
    print(f"  实体总数: {len(all_entities)}")
    
    if len(all_entities) == 0:
        print("警告: 未找到实体ID文件，将只统计三元组中出现的节点")
        all_entities = None
    
    # 计算节点度
    print("计算节点度...")
    degree_dict = compute_node_degrees(triples_1, triples_2, all_entities)
    
    # 保存结果
    output_file = os.path.join(dataset_path, 'node_degrees.txt')
    save_degree_dict(degree_dict, output_file)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compute node degrees (in+out) for each dataset')
    parser.add_argument('--base_dir', default='./data', help='Root directory containing the dataset folders')
    parser.add_argument('--datasets', nargs='+', default=['W-I-S1', 'W-I-S2', 'Y-I-S1', 'Y-I-S2'])
    args = parser.parse_args()

    base_path = args.base_dir
    datasets = args.datasets


    for dataset in datasets:
        dataset_path = os.path.join(base_path, dataset)
        if os.path.exists(dataset_path):
            process_dataset(dataset_path)
        else:
            print(f"\n数据集不存在: {dataset_path}")
    
    print(f"\n{'='*60}")
    print("所有数据集处理完成")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

