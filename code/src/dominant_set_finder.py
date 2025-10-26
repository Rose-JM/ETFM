import random
import os

class DominatingSetFinder:
    def __init__(self, cqag_file, properties_file):
        """
        初始化函数：加载图结构和图节点信息
        """
        self.graph = {}
        self.experts = set()
        self.qnum = 0
        self.anum = 0
        self.enum = 0
        self.N = 0

        self._load_properties(properties_file)
        self._load_graph(cqag_file)

    def _load_properties(self, properties_file):
        """
        加载节点基本统计信息
        """
        with open(properties_file, "r") as f:
            f.readline()  # skip header
            line = f.readline().strip().split()
            self.N = int(line[0])
            self.qnum = int(line[1])
            self.anum = int(line[2])
            self.enum = int(line[3])

    def _load_graph(self, cqag_file):
        """
        加载图的边（无向图），构建邻接表
        """
        with open(cqag_file, "r") as f:
            for line in f:
                u, v, w = map(int, line.strip().split())
                if u not in self.graph:
                    self.graph[u] = set()
                if v not in self.graph:
                    self.graph[v] = set()
                self.graph[u].add(v)
                self.graph[v].add(u)

        # 提取专家节点编号
        for i in range(self.qnum + self.anum, self.qnum + self.anum + self.enum):
            self.experts.add(i)

    def greedy_dominating_set(self):
        """
        使用贪婪算法生成初始支配集：优先选择覆盖未覆盖邻居最多的节点
        """
        uncovered = set(self.experts)
        dominating_set = set()

        while uncovered:
            max_cover = -1
            best_node = None
            for expert in self.experts:
                if expert in dominating_set:
                    continue
                neighbors = self.graph.get(expert, set())
                cover_count = len([n for n in neighbors if n in uncovered])
                if expert in uncovered:
                    cover_count += 1
                if cover_count > max_cover:
                    max_cover = cover_count
                    best_node = expert

            if best_node is None:
                break

            dominating_set.add(best_node)
            uncovered.discard(best_node)
            for neighbor in self.graph.get(best_node, set()):
                uncovered.discard(neighbor)

        return dominating_set

    def remove_redundant(self, ds):
        """
        尝试移除冗余节点（移除后仍能覆盖全图）
        """
        ds = list(ds)
        for expert in ds[:]:
            test_set = set(ds)
            test_set.remove(expert)

            covered = set()
            for e in test_set:
                covered.add(e)
                covered.update(self.graph.get(e, []))
            if self.experts.issubset(covered):
                ds.remove(expert)
        return set(ds)

    def random_perturbation(self, ds, max_iter=50):
        """
        引入扰动优化支配集（避免贪婪局部最优）
        """
        ds = list(ds)
        for _ in range(max_iter):
            i = random.randint(1, len(ds) - 1)
            ds.insert(0, ds.pop(i))
            new_set = self._rebuild_from_order(ds)
            if len(new_set) <= len(ds):
                ds = list(new_set)
        return set(ds)

    def _rebuild_from_order(self, order):
        """
        按指定顺序重新构造支配集（用于扰动）
        """
        uncovered = set(self.experts)
        dominating_set = set()

        for expert in order:
            if expert in uncovered:
                dominating_set.add(expert)
                uncovered.discard(expert)
                for neighbor in self.graph.get(expert, set()):
                    uncovered.discard(neighbor)
            if not uncovered:
                break
        return dominating_set

    def run(self):
        """
        执行整个流程：贪婪 -> 去冗余 -> 随机扰动
        """
        initial_set = self.greedy_dominating_set()
        reduced_set = self.remove_redundant(initial_set)
        optimized_set = self.random_perturbation(reduced_set)
        return optimized_set


if __name__ == "__main__":
    datasets = ["android", "history", "dba", "physics", "mathoverflow"]
    base_path = "/home/dyx2/team2box/team2box/data/"

    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        cqag_file = os.path.join(base_path, dataset, "CQAG.txt")
        properties_file = os.path.join(base_path, dataset, "CQAG_properties.txt")

        finder = DominatingSetFinder(cqag_file, properties_file)
        dominant_set = finder.run()

        output_path = os.path.join(base_path, dataset, "dominant_experts.txt")
        with open(output_path, "w") as f:
            for expert in sorted(dominant_set):
                f.write(str(expert) + "\n")

        print(f"Dominating set written to {output_path}")
