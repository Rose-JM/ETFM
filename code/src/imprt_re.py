import random
import os

# 设置文件路径
data_dir = "/home/dyx2/team2box/team2box/data/mathoverflow/"
input_file = os.path.join(data_dir, "answer_user_ids.txt")
output_dir = os.path.join(data_dir, "NeRankFormat")
output_file = os.path.join(output_dir, "results.txt")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

experts = set()

# 读取 answer_user_ids.txt 获取所有专家 ID
with open(input_file, "r") as fin:
    fin.readline()  # 跳过表头
    for line in fin:
        parts = line.strip().split()
        if len(parts) == 2:
            experts.add(parts[1])  # 只存专家 ID

experts = list(experts)  # 转换为列表

# 生成 results.txt，确保所有专家都有得分
with open(output_file, "w") as fout:
    scores = {eid: round(random.uniform(-5, 5), 4) for eid in experts}  # 生成随机得分
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)  # 按得分排序

    # 格式化输出
    result_line = " ".join([f"aid:{eid} score:{score}" for eid, score in sorted_scores])
    fout.write(result_line + "\n")

print(f"成功为 {len(experts)} 位专家生成评分，结果已保存到 {output_file}！")
