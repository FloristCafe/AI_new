import pandas as pd
import networkx as nx

# 1. 模拟：假设我们通过复杂的 NLP 句法树，已经提取出了“关系(边)”
# 真实场景下，这一步是你接下来几周在实验室要写的核心逻辑
triples_data = [
    {"主体": "张一鸣", "关系": "创办", "客体": "字节跳动"},
    {"主体": "李彦宏", "关系": "创办", "客体": "百度"},
    {"主体": "字节跳动", "关系": "竞争", "客体": "百度"},
    {"主体": "张一鸣", "关系": "注资", "客体": "大模型新计划"}
]
df_triples = pd.DataFrame(triples_data)

# 2. 实例化一个“有向图 (Directed Graph)”对象
# 这在底层就是为你开辟了一块由哈希表构成的极其高效的邻接表内存
G = nx.DiGraph()

# 3. 暴力建图：遍历矩阵，添加边和节点
for index, row in df_triples.iterrows():
    # add_edge() 会自动创建不存在的节点，并将它们连起来
    # 我们把“关系”作为边的一个属性（Attribute）存进去
    G.add_edge(row["主体"], row["客体"], relation=row["关系"])

print(f"========== 图谱实例化完成 ==========")
print(f"节点数量: {G.number_of_nodes()}")
print(f"边的数量: {G.number_of_edges()}")

print("\n========== 图结构底层探查 (C++ 算法映射) ==========")
# 看看“张一鸣”这个节点指向了谁（类似 C++ 里的遍历邻接表）
print(f"张一鸣的出度邻居: {list(G.successors('张一鸣'))}")

# 计算节点的“度中心性 (Degree Centrality)”
# 在量化金融里，如果这张图是公司股权图，中心度最高的往往是真正的财阀核心
centrality = nx.degree_centrality(G)
print(f"百度在网络中的中心度打分: {centrality['百度']:.2f}")