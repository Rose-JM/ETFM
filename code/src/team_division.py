import os
from collections import deque, defaultdict

class TeamDivider:
    def __init__(self, data_path, dataset, N):
        self.data_path = os.path.join(data_path, dataset)
        self.N = N
        self.graph = self.load_graph()                        # 加载图结构
        self.qnum, self.anum, self.enum = self.load_properties()  # 加载图节点数量
        self.dominant_experts = self.load_dominant_experts() # 加载支配集专家作为种子节点

    def load_graph(self):
        """加载图结构 CQAG.txt"""
        graph = defaultdict(list)
        with open(os.path.join(self.data_path, "CQAG.txt"), "r") as f:
            for line in f:
                u, v, w = map(int, line.strip().split())
                graph[u].append(v)
                graph[v].append(u)
        return graph

    def load_properties(self):
        """加载 CQAG_properties.txt"""
        with open(os.path.join(self.data_path, "CQAG_properties.txt"), "r") as f:
            f.readline()
            line = f.readline().strip()
            N, qnum, anum, enum = map(int, line.split())
        return qnum, anum, enum

    def load_dominant_experts(self):
        """加载支配集文件 dominant_experts.txt"""
        dom_path = os.path.join(self.data_path, "dominant_experts.txt")
        with open(dom_path, "r") as f:
            return [int(line.strip()) for line in f if line.strip().isdigit()]

    def get_P_neighbors(self, v):
        """获取 e-q-e 路径下的专家邻居"""
        neighbors = set()
        for a in self.graph[v]:
            if self.qnum <= a < self.qnum + self.anum:
                for e in self.graph[a]:
                    if e != v and e >= self.qnum + self.anum:
                        neighbors.add(e)
        return neighbors

    def get_all_P_neighbors(self, v):
        """扩展阶段弱连接邻居（P-N-N）"""
        return self.get_P_neighbors(v)

    def run_team_division(self):
        """执行团队划分过程"""
        results = dict()
        for seed in self.dominant_experts:
            Q = deque([seed])
            S = set([seed])
            D = deque()
            Visit = set([seed])
            Psi = defaultdict(set)

            while Q:
                v = Q.popleft()
                neighbors = self.get_P_neighbors(v)
                for u in neighbors:
                    if u not in Visit:
                        Psi[v].add(u)
                        Visit.add(u)
                S.update(Psi[v])
                if len(Psi[v]) >= self.N:
                    Q.extend(Psi[v])
                else:
                    D.append(v)

            while D:
                v = D.popleft()
                if v in S:
                    S.remove(v)
                for u in Psi[v]:
                    if u in S:
                        Psi[u].discard(v)
                        if len(Psi[u]) < self.N:
                            D.append(u)

            E = self.get_all_P_neighbors(seed)
            S.update(E)

            results[seed] = sorted(S)
        return results

    def save_teams(self, team_dict, save_path):
        """保存每个种子节点对应的团队成员列表"""
        with open(save_path, "w") as f:
            for seed, team in team_dict.items():
                f.write(f"{seed}: {' '.join(map(str, team))}\n")

if __name__ == "__main__":
    dataset_list = ["android", "history", "dba", "physics", "mathoverflow"]
    base_path = "/home/dyx2/team2box/team2box/data/"
    team_size_constraint = 3  # 设置约束 N

    for dataset in dataset_list:
        print(f"Processing dataset: {dataset}")
        divider = TeamDivider(base_path, dataset, team_size_constraint)
        teams = divider.run_team_division()
        output_path = os.path.join(base_path, dataset, "teams.txt")
        divider.save_teams(teams, output_path)
        print(f"Saved teams to {output_path}")
