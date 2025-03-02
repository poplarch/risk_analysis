# -*- coding: utf-8 -*-
import json
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import networkx as nx


class DecisionTreeRiskAnalyzer:
    """决策树风险分析器，用于评估风险连锁反应与复合影响"""

    def __init__(self):
        """初始化决策树风险分析器"""
        self.risk_tree = {}
        self.risk_paths = {}
        self.critical_paths = []

    def build_risk_tree(self, risk_dependencies: Dict[str, List[str]],
                        risk_probabilities: Dict[str, float],
                        risk_impacts: Dict[str, float]) -> Dict:
        """
        构建风险决策树
        
        Args:
            risk_dependencies (Dict[str, List[str]]): 风险依赖关系 {风险ID: [触发的后续风险ID列表]}
            risk_probabilities (Dict[str, float]): 风险发生概率 {风险ID: 概率值}
            risk_impacts (Dict[str, float]): 风险影响程度 {风险ID: 影响值}
            
        Returns:
            Dict: 风险树结构
        """
        self.risk_tree = {
            "dependencies": risk_dependencies,
            "probabilities": risk_probabilities,
            "impacts": risk_impacts,
            "nodes": set(risk_probabilities.keys())  # 所有风险节点集合
        }

        # 识别根节点（无前驱节点的风险）
        self.risk_tree["root_nodes"] = self._identify_root_nodes()

        # 为每个节点计算后继节点
        self.risk_tree["successors"] = self._calculate_successors()

        return self.risk_tree

    def _identify_root_nodes(self) -> List[str]:
        """
        识别风险树中的根节点（无前驱节点的风险）
        
        Returns:
            List[str]: 根节点风险ID列表
        """
        # 获取所有被依赖的节点
        dependent_nodes = set()
        for dependents in self.risk_tree["dependencies"].values():
            dependent_nodes.update(dependents)

        # 根节点是所有节点中不被其他节点依赖的节点
        root_nodes = self.risk_tree["nodes"] - dependent_nodes
        return list(root_nodes)

    def _calculate_successors(self) -> Dict[str, List[str]]:
        """
        计算每个风险节点的后继节点
        
        Returns:
            Dict[str, List[str]]: 后继节点映射 {风险ID: [后继风险ID列表]}
        """
        successors = {node: [] for node in self.risk_tree["nodes"]}

        for node, dependents in self.risk_tree["dependencies"].items():
            for dependent in dependents:
                if dependent in successors:
                    successors[node].append(dependent)

        return successors

    def calculate_chain_probability(self, risk_id: str, visited: Set[str] = None) -> float:
        """
        计算风险连锁反应的概率
        
        Args:
            risk_id (str): 风险ID
            visited (Set[str]): 已访问的风险集合，用于避免循环依赖
            
        Returns:
            float: 风险链条的条件概率
        """
        if visited is None:
            visited = set()

        # 防止循环依赖
        if risk_id in visited:
            return 0.0

        visited.add(risk_id)

        # 获取当前风险的概率
        base_probability = self.risk_tree["probabilities"].get(risk_id, 0.0)

        # 如果没有后继风险，直接返回当前风险概率
        successors = self.risk_tree["dependencies"].get(risk_id, [])
        if not successors:
            return base_probability

        # 计算后继风险的条件概率总和
        chain_probability = base_probability
        for successor in successors:
            successor_prob = self.calculate_chain_probability(successor, visited.copy())
            # 条件概率：后继风险概率 * 当前风险概率
            chain_probability += base_probability * successor_prob

        return min(chain_probability, 1.0)  # 确保概率不超过1

    def identify_risk_paths(self, max_depth: int = 5) -> Dict[str, List[List[str]]]:
        """
        识别从根节点到各风险节点的所有可能路径
        
        Args:
            max_depth (int): 最大路径深度，防止过深搜索
            
        Returns:
            Dict[str, List[List[str]]]: 每个节点的所有可能路径
        """
        self.risk_paths = {}

        for root in self.risk_tree["root_nodes"]:
            self._dfs_paths(root, [root], {}, max_depth)

        return self.risk_paths

    def _dfs_paths(self, current: str, path: List[str],
                   visited: Dict[str, bool], max_depth: int):
        """
        深度优先搜索查找所有路径
        
        Args:
            current (str): 当前节点
            path (List[str]): 当前路径
            visited (Dict[str, bool]): 访问记录
            max_depth (int): 最大深度
        """
        # 将路径添加到当前节点的路径集合中
        if current not in self.risk_paths:
            self.risk_paths[current] = []
        self.risk_paths[current].append(path.copy())

        # 如果达到最大深度，停止搜索
        if len(path) >= max_depth:
            return

        # 临时标记当前节点为已访问
        visited[current] = True

        # 遍历所有后继节点
        for successor in self.risk_tree["dependencies"].get(current, []):
            # 检查是否会形成环
            if successor in path:
                continue

            # 继续DFS
            self._dfs_paths(successor, path + [successor], visited.copy(), max_depth)

        # 回溯，移除当前节点的访问标记
        visited[current] = False

    def calculate_path_risk_values(self) -> Dict[str, Dict[str, float]]:
        """
        计算每条风险路径的风险值（概率×影响）
        
        Returns:
            Dict[str, Dict[str, float]]: 路径风险值 {路径ID: {"probability": p, "impact": i, "risk_value": r}}
        """
        path_values = {}

        # 确保已计算风险路径
        if not self.risk_paths:
            self.identify_risk_paths()

        # 为所有终端节点计算路径风险值
        for node, paths in self.risk_paths.items():
            for i, path in enumerate(paths):
                path_id = f"{node}_path{i + 1}"

                # 计算路径概率（条件概率链）
                path_prob = self.risk_tree["probabilities"].get(path[0], 0.0)
                for j in range(1, len(path)):
                    path_prob *= self.risk_tree["probabilities"].get(path[j], 0.0)

                # 计算复合影响（路径上所有风险的影响总和）
                # 使用加权求和，越靠后的风险影响越大
                path_impact = 0.0
                for j, risk in enumerate(path):
                    # 权重因子，使后续风险影响更重要
                    weight = 1.0 + 0.1 * j
                    path_impact += self.risk_tree["impacts"].get(risk, 0.0) * weight

                # 计算风险值
                risk_value = path_prob * path_impact

                path_values[path_id] = {
                    "path": path,
                    "probability": path_prob,
                    "impact": path_impact,
                    "risk_value": risk_value
                }

        return path_values

    def identify_critical_paths(self, top_n: int = 5) -> List[Dict]:
        """
        识别关键风险路径（风险值最高的路径）
        
        Args:
            top_n (int): 返回的关键路径数量
            
        Returns:
            List[Dict]: 关键路径列表，按风险值降序排列
        """
        # 计算所有路径的风险值
        path_values = self.calculate_path_risk_values()

        # 按风险值排序
        sorted_paths = sorted(
            path_values.items(),
            key=lambda x: x[1]["risk_value"],
            reverse=True
        )

        # 提取top_n路径
        self.critical_paths = [
            {
                "path_id": path_id,
                "risks": path_info["path"],
                "probability": path_info["probability"],
                "impact": path_info["impact"],
                "risk_value": path_info["risk_value"]
            }
            for path_id, path_info in sorted_paths[:top_n]
        ]

        return self.critical_paths

    def visualize_risk_tree(self, output_path: str, highlight_paths: List[List[str]] = None):
        """
        可视化风险决策树
        
        Args:
            output_path (str): 输出文件路径
            highlight_paths (List[List[str]], optional): 需要高亮显示的路径
        """
        # 创建有向图
        G = nx.DiGraph()

        # 添加节点
        for node in self.risk_tree["nodes"]:
            # 计算节点大小（基于风险影响）
            node_size = self.risk_tree["impacts"].get(node, 0.5) * 1000
            # 计算节点颜色（基于风险概率）
            node_color = self.risk_tree["probabilities"].get(node, 0.5)

            G.add_node(node, size=node_size, color=node_color)

        # 添加边
        for node, dependents in self.risk_tree["dependencies"].items():
            for dependent in dependents:
                G.add_edge(node, dependent)

        # 使用spring布局
        pos = nx.spring_layout(G, seed=42)

        plt.figure(figsize=(14, 10))

        # 绘制节点
        node_sizes = [G.nodes[node].get('size', 300) for node in G.nodes()]
        node_colors = [G.nodes[node].get('color', 0.5) for node in G.nodes()]

        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=plt.cm.Reds,
            alpha=0.8
        )

        # 绘制边
        edges = nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            arrows=True,
            arrowstyle='-|>',
            arrowsize=15,
            width=1.5,
            alpha=0.7
        )

        # 如果有需要高亮的路径
        if highlight_paths:
            for i, path in enumerate(highlight_paths):
                # 为每条路径选择不同的颜色
                path_colors = ['blue', 'green', 'purple', 'orange', 'cyan']
                color = path_colors[i % len(path_colors)]

                # 高亮路径上的边
                for j in range(len(path) - 1):
                    if G.has_edge(path[j], path[j + 1]):
                        edge_path = [(path[j], path[j + 1])]
                        nx.draw_networkx_edges(
                            G, pos,
                            edgelist=edge_path,
                            edge_color=color,
                            width=3.0,
                            arrows=True,
                            arrowstyle='-|>',
                            arrowsize=20
                        )

        # 添加标签
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

        # 添加图例
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('风险概率')

        plt.title('风险决策树分析')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def export_analysis_results(self, output_path: str):
        """
        导出风险决策树分析结果
        
        Args:
            output_path (str): 输出JSON文件路径
        """
        # 确保已完成关键路径分析
        if not self.critical_paths:
            self.identify_critical_paths()

        # 创建结果字典
        results = {
            "risk_nodes": {
                node: {
                    "probability": self.risk_tree["probabilities"].get(node, 0),
                    "impact": self.risk_tree["impacts"].get(node, 0),
                    "chain_probability": self.calculate_chain_probability(node)
                }
                for node in self.risk_tree["nodes"]
            },
            "critical_paths": self.critical_paths,
            "root_risks": self.risk_tree["root_nodes"],
            "risk_dependencies": self.risk_tree["dependencies"]
        }

        # 导出为JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        return results


class EventTreeRiskAnalyzer:
    """事件树风险分析器，用于评估风险事件的不同演化路径与结果"""

    def __init__(self):
        """初始化事件树风险分析器"""
        self.event_tree = {}
        self.outcomes = {}

    def build_event_tree(self, initial_event: str,
                         branch_probabilities: Dict[str, Dict[str, float]],
                         outcome_impacts: Dict[str, float]) -> Dict:
        """
        构建事件树
        
        Args:
            initial_event (str): 初始事件
            branch_probabilities (Dict[str, Dict[str, float]]): 分支概率 
                {节点ID: {分支ID: 概率值}}
            outcome_impacts (Dict[str, float]): 结果影响值
                {结果ID: 影响值}
            
        Returns:
            Dict: 事件树结构
        """
        self.event_tree = {
            "initial_event": initial_event,
            "branch_probabilities": branch_probabilities,
            "outcome_impacts": outcome_impacts
        }

        return self.event_tree

    def calculate_outcome_probabilities(self) -> Dict[str, float]:
        """
        计算每个可能结果的概率
        
        Returns:
            Dict[str, float]: 结果概率 {结果ID: 概率值}
        """
        # 获取初始事件和分支概率
        initial_event = self.event_tree["initial_event"]
        branch_probs = self.event_tree["branch_probabilities"]

        # 初始化结果概率字典
        outcome_probs = {}

        # 使用DFS计算所有可能路径的概率
        self._calculate_path_probabilities(initial_event, 1.0, [], outcome_probs)

        return outcome_probs

    def _calculate_path_probabilities(self, current_node: str, current_prob: float,
                                      path: List[str], outcome_probs: Dict[str, float]):
        """
        递归计算路径概率
        
        Args:
            current_node (str): 当前节点
            current_prob (float): 当前路径概率
            path (List[str]): 当前路径
            outcome_probs (Dict[str, float]): 结果概率字典（输出参数）
        """
        # 将当前节点添加到路径
        path = path + [current_node]

        # 检查当前节点是否有分支
        if current_node in self.event_tree["branch_probabilities"]:
            # 遍历所有分支
            for branch, prob in self.event_tree["branch_probabilities"][current_node].items():
                # 计算新路径的概率
                new_prob = current_prob * prob
                # 递归计算
                self._calculate_path_probabilities(branch, new_prob, path, outcome_probs)
        else:
            # 如果没有分支，则当前节点是一个结果
            outcome_id = "_".join(path)
            outcome_probs[outcome_id] = current_prob

    def calculate_risk_values(self) -> Dict[str, Dict[str, float]]:
        """
        计算每个可能结果的风险值
        
        Returns:
            Dict[str, Dict[str, float]]: 结果风险值 
                {结果ID: {"probability": p, "impact": i, "risk_value": r}}
        """
        # 计算结果概率
        outcome_probs = self.calculate_outcome_probabilities()

        # 初始化结果风险值
        self.outcomes = {}

        # 计算每个结果的风险值
        for outcome_id, probability in outcome_probs.items():
            # 解析结果ID（路径）
            path = outcome_id.split("_")
            final_node = path[-1]

            # 获取结果影响值
            impact = self.event_tree["outcome_impacts"].get(final_node, 0.0)

            # 计算风险值
            risk_value = probability * impact

            self.outcomes[outcome_id] = {
                "path": path,
                "probability": probability,
                "impact": impact,
                "risk_value": risk_value
            }

        return self.outcomes

    def identify_critical_outcomes(self, top_n: int = 3) -> List[Dict]:
        """
        识别关键风险结果（风险值最高的结果）
        
        Args:
            top_n (int): 返回的关键结果数量
            
        Returns:
            List[Dict]: 关键结果列表，按风险值降序排列
        """
        # 计算所有结果的风险值
        if not self.outcomes:
            self.calculate_risk_values()

        # 按风险值排序
        sorted_outcomes = sorted(
            self.outcomes.items(),
            key=lambda x: x[1]["risk_value"],
            reverse=True
        )

        # 提取top_n结果
        critical_outcomes = [
            {
                "outcome_id": outcome_id,
                "path": outcome_info["path"],
                "probability": outcome_info["probability"],
                "impact": outcome_info["impact"],
                "risk_value": outcome_info["risk_value"]
            }
            for outcome_id, outcome_info in sorted_outcomes[:top_n]
        ]

        return critical_outcomes

    def visualize_event_tree(self, output_path: str, highlight_outcomes: List[str] = None):
        """
        可视化事件树
        
        Args:
            output_path (str): 输出文件路径
            highlight_outcomes (List[str], optional): 需要高亮显示的结果ID
        """
        # 创建有向图
        G = nx.DiGraph()

        # 添加初始事件
        initial_event = self.event_tree["initial_event"]
        G.add_node(initial_event, type='initial')

        # 遍历所有分支
        for node, branches in self.event_tree["branch_probabilities"].items():
            # 添加当前节点
            if node != initial_event:
                G.add_node(node, type='branch')

            # 添加所有分支及连接
            for branch, prob in branches.items():
                G.add_node(branch, type='branch')
                G.add_edge(node, branch, probability=prob)

        # 添加结果节点
        for outcome, impact in self.event_tree["outcome_impacts"].items():
            if outcome not in G:
                G.add_node(outcome, type='outcome', impact=impact)

        # 计算结果路径（如果需要高亮）
        if highlight_outcomes:
            if not self.outcomes:
                self.calculate_risk_values()

            highlight_paths = []
            for outcome_id in highlight_outcomes:
                if outcome_id in self.outcomes:
                    highlight_paths.append(self.outcomes[outcome_id]["path"])

        # 使用分层布局
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

        plt.figure(figsize=(16, 10))

        # 为不同类型的节点设置不同的颜色
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node].get('type', '')
            if node_type == 'initial':
                node_colors.append('lightgreen')
            elif node_type == 'outcome':
                impact = G.nodes[node].get('impact', 0.5)
                # 根据影响程度设置颜色深浅
                if impact > 0.7:
                    node_colors.append('darkred')
                elif impact > 0.4:
                    node_colors.append('red')
                else:
                    node_colors.append('salmon')
            else:
                node_colors.append('lightblue')

        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.8)

        # 绘制边
        edges = nx.draw_networkx_edges(
            G, pos,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=15,
            width=1.5,
            alpha=0.7
        )

        # 如果有需要高亮的路径
        if highlight_outcomes:
            for i, path in enumerate(highlight_paths):
                # 为每条路径选择不同的颜色
                path_colors = ['blue', 'green', 'purple', 'orange', 'cyan']
                color = path_colors[i % len(path_colors)]

                # 高亮路径上的边
                for j in range(len(path) - 1):
                    if G.has_edge(path[j], path[j + 1]):
                        edge_path = [(path[j], path[j + 1])]
                        nx.draw_networkx_edges(
                            G, pos,
                            edgelist=edge_path,
                            edge_color=color,
                            width=3.0,
                            arrows=True,
                            arrowstyle='-|>',
                            arrowsize=20
                        )

        # 添加边标签（概率）
        edge_labels = {(u, v): f"{G[u][v]['probability']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

        # 添加节点标签
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

        plt.title('事件树风险分析')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def export_analysis_results(self, output_path: str):
        """
        导出事件树分析结果
        
        Args:
            output_path (str): 输出JSON文件路径
        """
        # 确保已计算结果
        if not self.outcomes:
            self.calculate_risk_values()

        # 获取关键结果
        critical_outcomes = self.identify_critical_outcomes()

        # 创建结果字典
        results = {
            "initial_event": self.event_tree["initial_event"],
            "outcomes": self.outcomes,
            "critical_outcomes": critical_outcomes,
            "expected_risk_value": sum(outcome["risk_value"] for outcome in self.outcomes.values())
        }

        # 导出为JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        return results
