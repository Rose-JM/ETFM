import os
import random

"""
脚本会遍历这些数据集，对每个数据集中的 knrmformat/train.txt 文件进行处理，并将结果保存为 knrmformat/train_cleaned.txt
"""
def clean_and_generate_negatives(input_file_path, output_file_path, vocab_size=2000000, min_len=5, max_len=100):
    """
    input_file_path: 原始 train.txt 路径
    output_file_path: 清洗后输出的路径
    vocab_size: 词表大小，用于生成随机负样本
    min_len: 负样本文档的最小长度
    max_len: 负样本文档的最大长度
    """
    if not os.path.exists(input_file_path):
        print(f"Input file {input_file_path} does not exist.")
        return

    with open(input_file_path, 'r', encoding='utf-8') as fin, open(output_file_path, 'w', encoding='utf-8') as fout:
        line_count = 0
        fix_count = 0

        for line in fin:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            parts = line.split(',')
            if len(parts) < 3:
                print(f"Warning: line {line_count+1} format error, skipped.")
                continue

            query_tokens = parts[0].strip()
            pos_doc_tokens = parts[1].strip()
            label_part = parts[2].strip()

            # 只取第一个标签
            if ',' in label_part:
                first_label = label_part.split(',')[0].strip()
            else:
                first_label = label_part

            # 随机生成负样本doc
            neg_doc_len = random.randint(min_len, max_len)
            neg_doc_tokens = ','.join(str(random.randint(1, vocab_size-1)) for _ in range(neg_doc_len))

            # 四列，Tab分隔写入
            fout.write(f"{query_tokens}\t{pos_doc_tokens}\t{neg_doc_tokens}\t1\n")
            line_count += 1

    print(f"Processed {line_count} lines.")
    print(f"Fixed {fix_count} lines with multiple labels.")
    print(f"Cleaned and negative-sampled file saved as {output_file_path}.")

if __name__ == "__main__":
    data = ["android", "dba", "physics", "mathoverflow", "history"]
    base_path = "/home/dyx2/team2box/team2box/data/"
    for d in data:
        input_path = os.path.join(base_path, d, "knrmformat", "train.txt")  # 加上 knrmformat 目录
        output_path = os.path.join(base_path, d, "knrmformat", "train_cleaned.txt")
        clean_and_generate_negatives(input_path, output_path)
