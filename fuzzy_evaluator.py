# -*- coding: utf-8 -*-
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

class FuzzyEvaluator:
    """模糊综合评价器，支持动态风险等级和隶属度函数"""
    def __init__(self, risk_levels: List[str] = None, membership_params: Optional[Dict[str, Tuple[float, float, float, float]]] = None):
        """初始化模糊评价器

        Args:
            risk_levels (List[str], optional): 风险等级列表，默认 ["VL", "L", "M", "H", "VH"]
            membership_params (Dict[str, Tuple[float, float, float, float]], optional): 隶属度函数参数，默认梯形函数
        """
        self.risk_levels = risk_levels or ["VL", "L", "M", "H", "VH"]
        self.membership_functions = self._create_membership_functions(membership_params or {
            "VL": (0, 0, 0.1, 0.3), "L": (0.1, 0.3, 0.4, 0.6), "M": (0.4, 0.5, 0.7, 0.8),
            "H": (0.7, 0.8, 0.9, 1.0), "VH": (0.9, 1.0, 1.2, 1.2)
        })

    def _create_membership_functions(self, params: Dict[str, Tuple[float, float, float, float]]) -> Dict[str, callable]:
        """创建梯形隶属度函数

        Args:
            params (Dict[str, Tuple[float, float, float, float]]): 每个风险等级的梯形参数 (a, b, c, d)

        Returns:
            Dict[str, callable]: 隶属度函数字典
        """
        def create_trap_mf(a, b, c, d):
            def trap_mf(x):
                if b == a or d == c:
                    return 1.0 if b <= x <= c else 0.0
                return np.maximum(0, np.minimum(np.minimum((x - a) / (b - a + 1e-6), (d - x) / (d - c + 1e-6)), 1))
            return trap_mf
        return {level: create_trap_mf(*param) for level, param in params.items()}

    def calculate_membership_degree(self, expert_scores: np.ndarray) -> np.ndarray:
        """计算隶属度向量

        Args:
            expert_scores (np.ndarray): 专家评分数组（归一化后，范围 0-1）

        Returns:
            np.ndarray: 隶属度向量
        """
        scores = np.asarray(expert_scores, dtype=float)
        membership = np.zeros(len(self.risk_levels))
        for i, mf in enumerate(self.membership_functions.values()):
            membership[i] = np.mean([mf(score) for score in scores])
        return membership / np.sum(membership) if np.sum(membership) > 0 else membership

    def fuzzy_multiply(self, weight_vector: np.ndarray, membership_matrix: np.ndarray) -> np.ndarray:
        """加权模糊运算，计算综合隶属度

        Args:
            weight_vector (np.ndarray): 全局权重向量
            membership_matrix (np.ndarray): 隶属度矩阵

        Returns:
            np.ndarray: 综合隶属度向量
        """
        result = np.dot(weight_vector, membership_matrix)
        return result / np.sum(result) if np.sum(result) > 0 else result

def perform_fuzzy_evaluation(fuzzy_global_weights: Dict[str, float], fuzzy_excel_path: str, excel_handler) -> Optional[np.ndarray]:
    """执行模糊综合评价

    Args:
        fuzzy_global_weights (Dict[str, float]): 全局权重字典
        fuzzy_excel_path (str): 模糊评价 Excel 文件路径
        excel_handler: Excel 数据处理器实例

    Returns:
        Optional[np.ndarray]: 模糊评价结果向量，如出错则返回 None
    """
    try:
        fuzzy_evaluator = FuzzyEvaluator()
        scores_df, _ = excel_handler.read_expert_scores(fuzzy_excel_path)
        
        normalized_scores = scores_df / 10.0
        membership_matrix = np.zeros((len(scores_df), fuzzy_evaluator.risk_levels))
        for i, factor in enumerate(scores_df.index):
            membership_matrix[i] = fuzzy_evaluator.calculate_membership_degree(normalized_scores.loc[factor].values)
        
        weight_vector = np.array([fuzzy_global_weights.get(factor, 0) for factor in scores_df.index])
        fuzzy_result = fuzzy_evaluator.fuzzy_multiply(weight_vector, membership_matrix)

        return fuzzy_result
    except Exception as e:
        logging.error(f"模糊综合评价出错: {str(e)}")
        print(f"模糊综合评价出错: {str(e)}")
        return None
