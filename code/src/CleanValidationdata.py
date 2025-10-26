import os

def clean_validation_file(input_file_path, output_file_path):
    if not os.path.exists(input_file_path):
        print(f"Input file {input_file_path} does not exist.")
        return

    with open(input_file_path, 'r', encoding='utf-8') as fin, open(output_file_path, 'w', encoding='utf-8') as fout:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 4:
                print(f"Warning: line {idx+1} format error, skipped.")
                continue

            query = parts[0].strip()
            pos_doc = parts[1].strip()
            neg_doc = parts[2].strip()
            label = parts[3].strip()

            # 只取第一个标签
            if ',' in label:
                label = label.split(',')[0].strip()

            # 按照Tab分隔重新写
            fout.write(f"{query}\t{pos_doc}\t{neg_doc}\t{label}\n")

    print(f"Validation file cleaned and saved to {output_file_path}")

if __name__ == "__main__":
    data = ["android", "dba", "physics", "mathoverflow", "history"]
    base_path = "/home/dyx2/team2box/team2box/data/"
    for d in data:
        input_path = os.path.join(base_path, d, "knrmformat", "validation.txt")  # 加上 knrmformat 目录
        output_path = os.path.join(base_path, d, "knrmformat", "validation_cleaned.txt")
        clean_validation_file(input_path, output_path)
