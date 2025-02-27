# -*- coding: utf-8 -*-
import numpy as np
import cvxpy as cp
from scipy.stats import gmean
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging

@dataclass
class ConsistencyResult:
    """一致性检查结果封装类，用于存储判断矩阵的一致性信息"""
    is_consistent: bool  # 是否一致（CR <= 0.1）
    consistency_ratio: float  # 一致性比率（CR）
    max_eigenvalue: float  # 最大特征值

@dataclass
class MatrixCorrectionResult:
    """矩阵修正结果封装类，用于存储修正前后的信息"""
    expert_id: int  # 专家编号
    original_cr: float  # 原始一致性比率
    final_cr: float  # 修正后一致性比率
    success: bool  # 是否满足一致性要求
    adjusted: bool  # 是否进行了修正

class ConsistencyChecker:
    """一致性检查器，用于验证 AHP 判断矩阵的一致性"""
    def __init__(self):
        # 随机一致性指数表，键为矩阵阶数，值为对应的 RI，用于计算一致性比率
        self.RI = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

    def check_consistency(self, matrix: np.ndarray) -> ConsistencyResult:
        """检查判断矩阵的一致性，返回一致性结果

        Args:
            matrix (np.ndarray): 输入的判断矩阵，需为方阵

        Returns:
            ConsistencyResult: 包含一致性比率 (CR)、最大特征值等结果

        Raises:
            ValueError: 如果矩阵不是方阵
        """
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("输入矩阵必须为方阵")
        n = len(matrix)
        eigenvalues, _ = np.linalg.eig(matrix)
        lambda_max = max(eigenvalues.real)
        CI = (lambda_max - n) / (n - 1) if n > 1 else 0
        RI = self.RI.get(n, 1.49)
        CR = CI / RI if RI != 0 else 0.0
        return ConsistencyResult(is_consistent=CR <= 0.1, consistency_ratio=CR, max_eigenvalue=lambda_max)

class MatrixValidator:
    """判断矩阵验证器，用于检查矩阵的有效性"""
    def __init__(self, scale_values: List[float]):
        # 标度值列表，用于验证矩阵元素是否在有效范围内（如 1/9 到 9）
        self.scale_values = scale_values

    def validate_upper_triangular(self, matrix: np.ndarray, context: str) -> None:
        """验证上三角矩阵的有效性，确保对角线为 1，上三角值在标度范围内

        Args:
            matrix (np.ndarray): 输入矩阵
            context (str): 验证上下文，用于错误信息

        Raises:
            ValueError: 如果矩阵不符合要求（非方阵、对角线非 1、值无效）
        """
        n = len(matrix)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{context} 中的矩阵必须为方阵")
        if not np.allclose(np.diag(matrix), 1):
            raise ValueError(f"{context} 中的对角线元素必须为 1")
        for i in range(n):
            for j in range(i + 1, n):
                if not any(np.isclose(matrix[i, j], v, atol=0.0001) for v in self.scale_values):
                    raise ValueError(f"{context} 中发现无效值 {matrix[i, j]}")

class WeightCalculator:
    """权重计算器，用于从判断矩阵计算优先级权重"""
    @staticmethod
    def calculate_weights(matrix: np.ndarray) -> np.ndarray:
        """使用特征向量法计算权重

        Args:
            matrix (np.ndarray): 判断矩阵

        Returns:
            np.ndarray: 归一化后的权重向量
        """
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        max_index = np.argmax(eigenvalues.real)
        weights = eigenvectors[:, max_index].real
        return weights / np.sum(weights)

class MatrixAggregator:
    """多专家判断矩阵聚合器，使用加权几何平均法"""
    def __init__(self, expert_weights: Optional[List[float]] = None):
        # 专家权重，如果未提供则默认均等分配
        self.expert_weights = expert_weights

    def aggregate_judgments(self, matrices: List[np.ndarray]) -> np.ndarray:
        """聚合多专家判断矩阵

        Args:
            matrices (List[np.ndarray]): 多专家判断矩阵列表

        Returns:
            np.ndarray: 聚合后的判断矩阵

        Raises:
            ValueError: 如果矩阵列表为空
        """
        if not matrices:
            raise ValueError("专家判断矩阵列表不能为空")
        if self.expert_weights is None:
            self.expert_weights = [1 / len(matrices)] * len(matrices)
        n = len(matrices[0])
        aggregated = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                values = [matrix[i][j] for matrix in matrices]
                aggregated[i][j] = gmean(values, weights=self.expert_weights)
        return aggregated

class MatrixCorrector:
    """不一致判断矩阵修正器，使用对数最小二乘法 (LLSM)"""
    def __init__(self, scale_values: List[float]):
        self.scale_values = scale_values
        self.consistency_checker = ConsistencyChecker()

    def project_to_scale(self, value: float) -> float:
        """将值投影到最近的有效标度值"""
        scale_arr = np.array(self.scale_values)
        diffs = np.abs(scale_arr - value)
        return float(scale_arr[np.argmin(diffs)])

    def llsm_adjustment_with_projection(self, matrix: np.ndarray, penalty_weight: float = 1e-3) -> Tuple[np.ndarray, bool, float, bool]:
        """使用对数最小二乘法修正不一致矩阵"""
        CR = self.consistency_checker.check_consistency(matrix).consistency_ratio
        if CR < 0.1:
            return matrix, True, CR, False
        n = matrix.shape[0]
        x = cp.Variable(n)
        constraints = [x[0] == 0]
        obj_expr = sum(cp.square(cp.log(matrix[i, j]) - (x[i] - x[j])) for i in range(n) for j in range(n))
        penalty = penalty_weight * cp.sum_squares(x)
        objective = cp.Minimize(obj_expr + penalty)
        prob = cp.Problem(objective, constraints)
        prob.solve()
        w = np.exp(x.value)
        w = w / np.sum(w)
        A_intermediate = np.outer(w, 1 / w)
        A_final = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                proj_val = self.project_to_scale(float(A_intermediate[i, j]))
                A_final[i, j] = proj_val
                A_final[j, i] = 1.0 / proj_val
        final_consistency_result = self.consistency_checker.check_consistency(A_final)
        return A_final, final_consistency_result.is_consistent, final_consistency_result.consistency_ratio, True

class AHPProcessor:
    """AHP 处理模块，负责层次分析法的计算和处理"""
    def __init__(self, excel_prefix: str, expert_weights: Optional[List[float]] = None):
        """初始化 AHP 处理器

        Args:
            excel_prefix (str): AHP Excel 文件前缀
            expert_weights (Optional[List[float]]): 专家权重列表
        """
        self.excel_prefix = excel_prefix
        self.expert_weights = expert_weights
        self.scale_values = [1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.matrix_validator = MatrixValidator(self.scale_values)
        self.ahp_system = RiskAHPSystem(expert_weights)

    def process_level(self, level: str) -> Dict:
        """处理单个层次的 AHP 数据

        Args:
            level (str): 层级名称（如 'Goal', '技术风险C1'）

        Returns:
            Dict: AHP 处理结果，包括权重和一致性信息
        """
        excel_path = f"{self.excel_prefix}_{level.replace(' ', '')}.xlsx"
        matrices, criteria_names = self.ahp_system.excel_handler.read_expert_matrices(excel_path)
        return self.ahp_system.process_expert_judgments(matrices, criteria_names)

class RiskAHPSystem:
    """AHP 系统核心类，协调一致性检查、矩阵修正和权重计算"""
    def __init__(self, expert_weights: Optional[List[float]] = None):
        self.scale_values = [1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.matrix_validator = MatrixValidator(self.scale_values)
        self.consistency_checker = ConsistencyChecker()
        self.weight_calculator = WeightCalculator()
        self.matrix_aggregator = MatrixAggregator(expert_weights)
        self.matrix_corrector = MatrixCorrector(self.scale_values)

    def process_expert_judgments(self, matrices: List[np.ndarray], criteria_names: Optional[List[str]] = None) -> Dict:
        """处理专家判断矩阵，计算权重和一致性"""
        if not matrices or not all(isinstance(m, np.ndarray) for m in matrices):
            raise ValueError("输入矩阵列表无效")
        n = len(matrices[0])
        if not criteria_names:
            criteria_names = [f"准则 {i + 1}" for i in range(n)]
        original_matrices = []
        corrected_matrices = []
        correction_results = []
        for i, matrix in enumerate(matrices):
            original_CR = self.consistency_checker.check_consistency(matrix).consistency_ratio
            corrected_matrix, is_success, final_CR, is_adjusted = self.matrix_corrector.llsm_adjustment_with_projection(matrix)
            original_matrices.append(matrix)
            corrected_matrices.append(corrected_matrix)
            correction_results.append(MatrixCorrectionResult(expert_id=i + 1, original_cr=round(original_CR, 4), final_cr=round(final_CR, 4), success=is_success, adjusted=is_adjusted))
        aggregated_matrix = self.matrix_aggregator.aggregate_judgments(corrected_matrices)
        final_weights = self.weight_calculator.calculate_weights(aggregated_matrix)
        aggregated_consistency_result = self.consistency_checker.check_consistency(aggregated_matrix)
        return {
            "correction_results": correction_results,
            "original_matrices": original_matrices,
            "corrected_matrices": corrected_matrices,
            "aggregated_matrix": aggregated_matrix,
            "final_weights": dict(zip(criteria_names, final_weights)),
            "aggregated_consistency": aggregated_consistency_result
        }
