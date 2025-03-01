# -*- coding: utf-8 -*-
"""
优化的层次分析法(AHP)处理模块
适用于多专家群策决策场景下的层次分析法实现
支持判断矩阵一致性检验、自动修正与结果聚合
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import cvxpy as cp
import numpy as np
from scipy.stats import gmean


@dataclass
class ConsistencyResult:
    """一致性检查结果封装类，用于存储判断矩阵的一致性信息"""
    is_consistent: bool  # 是否一致（CR <= 0.1）
    consistency_ratio: float  # 一致性比率（CR）
    max_eigenvalue: float  # 最大特征值
    consistency_index: float = field(default=0.0)  # 一致性指标（CI）
    random_index: float = field(default=0.0)  # 随机一致性指标（RI）


@dataclass
class MatrixCorrectionResult:
    """矩阵修正结果封装类，用于存储修正前后的信息"""
    expert_id: int  # 专家编号
    original_cr: float  # 原始一致性比率
    final_cr: float  # 修正后一致性比率
    success: bool  # 是否满足一致性要求
    adjusted: bool  # 是否进行了修正
    correction_method: str = field(default="LLSM")  # 使用的修正方法
    iterations: int = field(default=0)  # 修正迭代次数


class ConsistencyChecker:
    """一致性检查器，用于验证 AHP 判断矩阵的一致性"""

    def __init__(self):
        """初始化一致性检查器"""
        # 随机一致性指数表，键为矩阵阶数，值为对应的 RI，用于计算一致性比率
        self.RI = {
            1: 0.00, 2: 0.00, 3: 0.58, 4: 0.9, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
            11: 1.51, 12: 1.54, 13: 1.56, 14: 1.57, 15: 1.58
        }

        # 配置日志
        self.logger = logging.getLogger(__name__)

    def check_consistency(self, matrix: np.ndarray) -> ConsistencyResult:
        """
        检查判断矩阵的一致性，返回一致性结果

        Args:
            matrix (np.ndarray): 输入的判断矩阵，需为方阵

        Returns:
            ConsistencyResult: 包含一致性比率 (CR)、最大特征值等结果

        Raises:
            ValueError: 如果矩阵不是方阵或无法计算特征值
        """
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("输入矩阵必须为方阵")

        n = len(matrix)

        try:
            # 计算特征值
            eigenvalues, _ = np.linalg.eig(matrix)
            lambda_max = max(eigenvalues.real)

            # 计算一致性指标 CI = (λmax - n) / (n - 1)
            CI = (lambda_max - n) / (n - 1) if n > 1 else 0

            # 查找随机一致性指标 RI
            RI = self.RI.get(n, 1.59)  # 默认使用 1.59 作为大于 15 阶的 RI 近似值

            # 计算一致性比率 CR = CI / RI
            CR = CI / RI if RI != 0 else 0.0

            # 确定是否一致 (CR <= 0.1 视为通过一致性检验)
            is_consistent = CR <= 0.1

            self.logger.debug(f"矩阵一致性检查: 阶数={n}, λmax={lambda_max:.4f}, "
                              f"CI={CI:.4f}, RI={RI:.4f}, CR={CR:.4f}, "
                              f"一致性: {'通过' if is_consistent else '不通过'}")

            return ConsistencyResult(
                is_consistent=is_consistent,
                consistency_ratio=CR,
                max_eigenvalue=lambda_max,
                consistency_index=CI,
                random_index=RI
            )

        except Exception as e:
            self.logger.error(f"计算一致性时出错: {str(e)}")
            raise ValueError(f"计算矩阵特征值出错: {str(e)}")


class MatrixValidator:
    """判断矩阵验证器，用于检查矩阵的有效性"""

    def __init__(self, scale_values: Optional[List[float]] = None):
        """
        初始化判断矩阵验证器

        Args:
            scale_values (Optional[List[float]]): 标度值列表，用于验证矩阵元素
                                                是否在有效范围内（如 1/9 到 9）
        """
        # 标度值列表，用于验证矩阵元素是否在有效范围内
        self.scale_values = scale_values or self._generate_default_scale_values()
        self.logger = logging.getLogger(__name__)

    def _generate_default_scale_values(self) -> List[float]:
        """
        生成默认的标度值列表

        Returns:
            List[float]: 默认标度值列表，包含 1/9 到 9 的标度值
        """
        # 生成标准 AHP 标度值和倒数
        integer_scales = list(range(1, 10))  # 1-9
        reciprocals = [1 / x for x in integer_scales[1:]]  # 1/2-1/9
        return sorted(reciprocals + integer_scales)

    def validate_upper_triangular(self, matrix: np.ndarray, context: str = "") -> bool:
        """
        验证上三角矩阵的有效性，确保对角线为 1，上三角值在标度范围内

        Args:
            matrix (np.ndarray): 输入矩阵
            context (str): 验证上下文，用于错误信息

        Returns:
            bool: 验证是否通过

        Raises:
            ValueError: 如果矩阵不符合要求（非方阵、对角线非 1、值无效）
        """
        # 确保矩阵是方阵
        n = len(matrix)
        if matrix.shape[0] != matrix.shape[1]:
            error_msg = f"{context} 中的矩阵必须为方阵"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # 验证对角线元素是否都为 1
        if not np.allclose(np.diag(matrix), 1, atol=1e-5):
            error_msg = f"{context} 中的对角线元素必须为 1"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # 验证上三角元素是否都在有效标度值范围内
        for i in range(n):
            for j in range(i + 1, n):
                value = matrix[i, j]
                if not any(np.isclose(value, v, atol=1e-5) for v in self.scale_values):
                    error_msg = f"{context} 中发现无效值 {value} 在位置 ({i + 1}, {j + 1})"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
        return True

    def validate_matrix(self, matrix: np.ndarray, context: str = "") -> bool:
        """
        验证完整判断矩阵的有效性

        Args:
            matrix (np.ndarray): 输入矩阵
            context (str): 验证上下文，用于错误信息

        Returns:
            bool: 验证是否通过

        Raises:
            ValueError: 如果矩阵不符合要求
        """
        # 确保矩阵是方阵
        n = len(matrix)
        if matrix.shape[0] != matrix.shape[1]:
            error_msg = f"{context} 中的矩阵必须为方阵"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # 验证对角线元素是否都为 1
        if not np.allclose(np.diag(matrix), 1, atol=1e-5):
            error_msg = f"{context} 中的对角线元素必须为 1"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # 验证互反性
        for i in range(n):
            for j in range(i + 1, n):
                if not np.isclose(matrix[j, i], 1.0 / matrix[i, j], atol=1e-5):
                    error_msg = f"{context} 中位置 ({j + 1}, {i + 1}) 的值应为 ({i + 1}, {j + 1}) 的倒数"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)

        return True


class WeightCalculator:
    """权重计算器，用于从判断矩阵计算优先级权重"""

    def __init__(self):
        """初始化权重计算器"""
        self.logger = logging.getLogger(__name__)
        self.available_methods = ["eigenvector", "geometric", "arithmetic"]

    def calculate_weights(self, matrix: np.ndarray, method: str = "eigenvector") -> np.ndarray:
        """
        计算判断矩阵的权重向量

        Args:
            matrix (np.ndarray): 判断矩阵
            method (str): 计算方法 'eigenvector'(特征向量法), 'geometric'(几何平均法),
                         或 'arithmetic'(算术平均法)

        Returns:
            np.ndarray: 归一化后的权重向量

        Raises:
            ValueError: 如果指定的方法不支持
        """
        if method not in self.available_methods:
            error_msg = f"不支持的权重计算方法: {method}，可选: {', '.join(self.available_methods)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            if method == "eigenvector":
                weights = self._eigenvector_method(matrix)
            elif method == "geometric":
                weights = self._geometric_mean_method(matrix)
            else:  # arithmetic
                weights = self._arithmetic_mean_method(matrix)

            # 确保权重之和为 1
            weights = weights / np.sum(weights)
            return weights

        except Exception as e:
            self.logger.error(f"计算权重时出错 (方法: {method}): {str(e)}")
            raise

    def _eigenvector_method(self, matrix: np.ndarray) -> np.ndarray:
        """
        使用特征向量法计算权重

        Args:
            matrix (np.ndarray): 判断矩阵

        Returns:
            np.ndarray: 归一化后的权重向量
        """
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(matrix)

        # 找到最大特征值对应的索引
        max_index = np.argmax(eigenvalues.real)

        # 提取对应的特征向量并取实部
        weights = np.abs(eigenvectors[:, max_index].real)

        return weights

    def _geometric_mean_method(self, matrix: np.ndarray) -> np.ndarray:
        """
        使用几何平均法计算权重

        Args:
            matrix (np.ndarray): 判断矩阵

        Returns:
            np.ndarray: 归一化后的权重向量
        """
        n = len(matrix)
        weights = np.zeros(n)

        # 计算每行的几何平均值
        for i in range(n):
            weights[i] = gmean(matrix[i, :])

        return weights

    def _arithmetic_mean_method(self, matrix: np.ndarray) -> np.ndarray:
        """
        使用算术平均法计算权重

        Args:
            matrix (np.ndarray): 判断矩阵

        Returns:
            np.ndarray: 归一化后的权重向量
        """
        # 将矩阵归一化（按列）
        col_sums = matrix.sum(axis=0)
        normalized_matrix = matrix / col_sums

        # 计算行平均值作为权重
        weights = normalized_matrix.mean(axis=1)

        return weights


class MatrixAggregator:
    """多专家判断矩阵聚合器，支持多种聚合方法"""

    def __init__(self, expert_weights: Optional[List[float]] = None):
        """
        初始化矩阵聚合器

        Args:
            expert_weights (Optional[List[float]]): 专家权重，如未提供则默认均等分配
        """
        self.expert_weights = expert_weights
        self.logger = logging.getLogger(__name__)
        self.available_methods = ["geometric", "arithmetic", "weighted"]

    def aggregate_judgments(self, matrices: List[np.ndarray],
                            method: str = "geometric") -> np.ndarray:
        """
        聚合多专家判断矩阵

        Args:
            matrices (List[np.ndarray]): 多专家判断矩阵列表
            method (str): 聚合方法 'geometric'(加权几何平均), 'arithmetic'(加权算术平均),
                         或 'weighted'(自定义加权)

        Returns:
            np.ndarray: 聚合后的判断矩阵

        Raises:
            ValueError: 如果矩阵列表为空或方法不支持
        """
        if not matrices:
            error_msg = "专家判断矩阵列表不能为空"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if method not in self.available_methods:
            error_msg = f"不支持的聚合方法: {method}，可选: {', '.join(self.available_methods)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # 检查所有矩阵的维度是否一致
        n = matrices[0].shape[0]
        for i, matrix in enumerate(matrices):
            if matrix.shape != (n, n):
                error_msg = f"第 {i + 1} 个矩阵的维度 {matrix.shape} 与第一个矩阵的维度 {(n, n)} 不一致"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        # 初始化专家权重（如果未提供则均等分配）
        if self.expert_weights is None:
            self.expert_weights = [1.0 / len(matrices)] * len(matrices)
            self.logger.debug(f"使用默认均等专家权重: {self.expert_weights}")

        # 确保权重数量与矩阵数量一致
        if len(self.expert_weights) != len(matrices):
            self.logger.warning(f"专家权重数量 ({len(self.expert_weights)}) 与矩阵数量 "
                                f"({len(matrices)}) 不一致，将调整为均等权重")
            self.expert_weights = [1.0 / len(matrices)] * len(matrices)

        # 确保权重和为 1
        weight_sum = sum(self.expert_weights)
        if not np.isclose(weight_sum, 1.0):
            self.logger.warning(f"专家权重和 ({weight_sum}) 不为 1，将进行归一化")
            self.expert_weights = [w / weight_sum for w in self.expert_weights]

        try:
            if method == "geometric":
                aggregated = self._weighted_geometric_mean(matrices)
            elif method == "arithmetic":
                aggregated = self._weighted_arithmetic_mean(matrices)
            else:  # weighted
                aggregated = self._custom_weighted_aggregation(matrices)

            # 确保对角线为 1
            np.fill_diagonal(aggregated, 1.0)

            # 确保互反性
            for i in range(n):
                for j in range(i + 1, n):
                    aggregated[j, i] = 1.0 / aggregated[i, j]

            return aggregated

        except Exception as e:
            self.logger.error(f"聚合判断矩阵时出错 (方法: {method}): {str(e)}")
            raise

    def _weighted_geometric_mean(self, matrices: List[np.ndarray]) -> np.ndarray:
        """
        使用加权几何平均法聚合矩阵

        Args:
            matrices (List[np.ndarray]): 多专家判断矩阵列表

        Returns:
            np.ndarray: 聚合后的判断矩阵
        """
        n = matrices[0].shape[0]
        aggregated = np.ones((n, n))

        # 对每个元素计算加权几何平均值
        for i in range(n):
            for j in range(n):
                if i != j:  # 跳过对角线元素
                    values = [matrix[i, j] for matrix in matrices]
                    aggregated[i, j] = gmean(values, weights=self.expert_weights)

        return aggregated

    def _weighted_arithmetic_mean(self, matrices: List[np.ndarray]) -> np.ndarray:
        """
        使用加权算术平均法聚合矩阵

        Args:
            matrices (List[np.ndarray]): 多专家判断矩阵列表

        Returns:
            np.ndarray: 聚合后的判断矩阵
        """
        n = matrices[0].shape[0]
        aggregated = np.zeros((n, n))

        # 对每个元素计算加权算术平均值
        for i in range(n):
            for j in range(n):
                values = [matrix[i, j] for matrix in matrices]
                aggregated[i, j] = np.average(values, weights=self.expert_weights)

        return aggregated

    def _custom_weighted_aggregation(self, matrices: List[np.ndarray]) -> np.ndarray:
        """
        使用自定义加权方法聚合矩阵

        Args:
            matrices (List[np.ndarray]): 多专家判断矩阵列表

        Returns:
            np.ndarray: 聚合后的判断矩阵
        """
        # 当前实现使用对数变换后的加权算术平均，可根据需要自定义
        n = matrices[0].shape[0]
        aggregated = np.ones((n, n))

        # 对每个元素进行对数变换后计算加权平均，然后取指数
        for i in range(n):
            for j in range(n):
                if i != j:  # 跳过对角线元素
                    log_values = [np.log(matrix[i, j]) for matrix in matrices]
                    log_avg = np.average(log_values, weights=self.expert_weights)
                    aggregated[i, j] = np.exp(log_avg)

        return aggregated


class MatrixCorrector:
    """不一致判断矩阵修正器，支持多种修正方法"""

    def __init__(self, scale_values: Optional[List[float]] = None):
        """
        初始化矩阵修正器

        Args:
            scale_values (Optional[List[float]]): 标度值列表，用于投影修正后的值
        """
        self.scale_values = scale_values or self._generate_default_scale_values()
        self.consistency_checker = ConsistencyChecker()
        self.weight_calculator = WeightCalculator()
        self.logger = logging.getLogger(__name__)
        self.available_methods = ["LLSM", "direct", "iterative"]

    def _generate_default_scale_values(self) -> List[float]:
        """
        生成默认的标度值列表

        Returns:
            List[float]: 默认标度值列表，包含 1/9 到 9 的标度值
        """
        # 生成标准 AHP 标度值和倒数
        integer_scales = list(range(1, 10))  # 1-9
        reciprocals = [1 / x for x in integer_scales[1:]]  # 1/2-1/9
        return sorted(reciprocals + integer_scales)

    def project_to_scale(self, value: float) -> float:
        """
        将值投影到最近的有效标度值

        Args:
            value (float): 输入值

        Returns:
            float: 最接近的有效标度值
        """
        if value <= 0:
            return min(self.scale_values)

        # 找到最接近的标度值
        scale_arr = np.array(self.scale_values)
        diffs = np.abs(scale_arr - value)
        closest_idx = np.argmin(diffs)

        return float(scale_arr[closest_idx])

    def correct_matrix(self, matrix: np.ndarray, method: str = "LLSM",
                       max_iterations: int = 10) -> Tuple[np.ndarray, bool, float, bool]:
        """
        修正不一致判断矩阵

        Args:
            matrix (np.ndarray): 原始判断矩阵
            method (str): 修正方法 'LLSM'(对数最小二乘法), 'direct'(直接投影法),
                         或 'iterative'(迭代修正法)
            max_iterations (int): 最大迭代次数（用于迭代方法）

        Returns:
            Tuple[np.ndarray, bool, float, bool]: 修正后矩阵, 是否一致, CR值, 是否经过修正

        Raises:
            ValueError: 如果指定的方法不支持
        """
        if method not in self.available_methods:
            error_msg = f"不支持的矩阵修正方法: {method}，可选: {', '.join(self.available_methods)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # 首先检查原始矩阵是否已经一致
        CR = self.consistency_checker.check_consistency(matrix).consistency_ratio
        if CR < 0.1:
            self.logger.debug(f"矩阵已满足一致性要求，CR={CR:.4f}，无需修正")
            return matrix, True, CR, False

        try:
            if method == "LLSM":
                result = self.llsm_adjustment_with_projection(matrix)
            elif method == "direct":
                result = self.direct_projection_adjustment(matrix)
            else:  # iterative
                result = self.iterative_adjustment(matrix, max_iterations)

            # 确保修正后的CR值是否小于原始CR值的某个百分比，才认为是有效修正
            corrected_matrix, is_consistent, final_CR, _ = result

            # 判断是否真正进行了修正（通过比较矩阵元素是否变化）
            is_actually_adjusted = not np.allclose(matrix, corrected_matrix, rtol=1e-5, atol=1e-8)

            return corrected_matrix, is_consistent, final_CR, is_actually_adjusted
        except Exception as e:
            self.logger.error(f"修正矩阵时出错 (方法: {method}): {str(e)}")
            raise

    def llsm_adjustment_with_projection(self, matrix: np.ndarray,
                                        penalty_weight: float = 1e-3) -> Tuple[np.ndarray, bool, float, bool]:
        """
        使用对数最小二乘法修正不一致矩阵

        Args:
            matrix (np.ndarray): 原始判断矩阵
            penalty_weight (float): 正则化惩罚权重

        Returns:
            Tuple[np.ndarray, bool, float, bool]: 修正后矩阵, 是否一致, CR值, 是否经过修正
        """
        n = matrix.shape[0]

        # 使用 CVXPY 求解对数最小二乘问题
        x = cp.Variable(n)
        constraints = [x[0] == 0]  # 固定第一个权重为基准

        # 构建目标函数：最小化对数差异平方和
        obj_expr = sum(cp.square(cp.log(matrix[i, j]) - (x[i] - x[j]))
                       for i in range(n) for j in range(n) if i != j)

        # 添加正则化项，防止数值不稳定
        penalty = penalty_weight * cp.sum_squares(x)
        objective = cp.Minimize(obj_expr + penalty)

        # 求解优化问题
        prob = cp.Problem(objective, constraints)
        prob.solve()

        if prob.status != cp.OPTIMAL:
            self.logger.warning(f"对数最小二乘优化未能找到最优解，状态: {prob.status}")
            return matrix, False, 1.0, False

        # 从优化结果计算权重向量
        w = np.exp(x.value)
        w = w / np.sum(w)

        # 使用权重构建一致矩阵
        A_intermediate = np.outer(w, 1 / w)

        # 投影到标度值
        A_final = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                proj_val = self.project_to_scale(float(A_intermediate[i, j]))
                A_final[i, j] = proj_val
                A_final[j, i] = 1.0 / proj_val

        # 检查修正后的一致性
        final_consistency_result = self.consistency_checker.check_consistency(A_final)

        return A_final, final_consistency_result.is_consistent, final_consistency_result.consistency_ratio, True

    def direct_projection_adjustment(self, matrix: np.ndarray) -> Tuple[np.ndarray, bool, float, bool]:
        """
        使用直接投影法修正不一致矩阵

        Args:
            matrix (np.ndarray): 原始判断矩阵

        Returns:
            Tuple[np.ndarray, bool, float, bool]: 修正后矩阵, 是否一致, CR值, 是否经过修正
        """
        n = matrix.shape[0]

        # 计算原始矩阵的权重向量
        w = self.weight_calculator.calculate_weights(matrix)

        # 使用权重构建一致矩阵
        A_intermediate = np.outer(w, 1 / w)

        # 投影到标度值
        A_final = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                proj_val = self.project_to_scale(float(A_intermediate[i, j]))
                A_final[i, j] = proj_val
                A_final[j, i] = 1.0 / proj_val

        # 检查修正后的一致性
        final_consistency_result = self.consistency_checker.check_consistency(A_final)

        return A_final, final_consistency_result.is_consistent, final_consistency_result.consistency_ratio, True

    def iterative_adjustment(self, matrix: np.ndarray,
                             max_iterations: int = 100) -> Tuple[np.ndarray, bool, float, bool]:
        """
        使用迭代修正法修正不一致判断矩阵

        Args:
            matrix (np.ndarray): 原始判断矩阵
            max_iterations (int): 最大迭代次数

        Returns:
            Tuple[np.ndarray, bool, float, bool]: 修正后矩阵, 是否一致, CR值, 是否经过修正
        """
        n = matrix.shape[0]
        A_modified = matrix.copy()

        # 迭代修正
        for iteration in range(max_iterations):
            # 计算当前矩阵的一致性
            consistency_result = self.consistency_checker.check_consistency(A_modified)
            if consistency_result.is_consistent:
                self.logger.debug(f"迭代修正成功，迭代次数: {iteration + 1}，"
                                  f"CR={consistency_result.consistency_ratio:.4f}")
                return A_modified, True, consistency_result.consistency_ratio, True

            # 计算当前矩阵的权重向量
            w = self.weight_calculator.calculate_weights(A_modified)

            # 基于权重构建完全一致矩阵
            A_consistent = np.outer(w, 1 / w)

            # 计算调整量
            delta = 0.3  # 调整因子，可根据需要调整
            A_adjusted = (1 - delta) * A_modified + delta * A_consistent

            # 确保对角线为1
            np.fill_diagonal(A_adjusted, 1.0)

            # 确保互反性
            for i in range(n):
                for j in range(i + 1, n):
                    A_adjusted[j, i] = 1.0 / A_adjusted[i, j]

            # 投影到标度值
            A_final = np.ones((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    proj_val = self.project_to_scale(float(A_adjusted[i, j]))
                    A_final[i, j] = proj_val
                    A_final[j, i] = 1.0 / proj_val

            # 更新矩阵
            A_modified = A_final

        # 最大迭代后的一致性检查
        final_result = self.consistency_checker.check_consistency(A_modified)
        self.logger.warning(f"达到最大迭代次数 {max_iterations}，"
                            f"最终 CR={final_result.consistency_ratio:.4f}")

        return A_modified, final_result.is_consistent, final_result.consistency_ratio, True


class AHPProcessor:
    """AHP 处理模块，负责层次分析法的计算和处理"""

    def __init__(self, excel_prefix: str, expert_weights: Optional[List[float]] = None):
        """
        初始化 AHP 处理器

        Args:
            excel_prefix (str): AHP Excel 文件前缀
            expert_weights (Optional[List[float]]): 专家权重列表
        """
        self.excel_prefix = excel_prefix
        self.expert_weights = expert_weights
        self.scale_values = self._generate_scale_values()
        self.matrix_validator = MatrixValidator(self.scale_values)
        self.consistency_checker = ConsistencyChecker()
        self.weight_calculator = WeightCalculator()
        self.matrix_aggregator = MatrixAggregator(expert_weights)
        self.matrix_corrector = MatrixCorrector(self.scale_values)
        self.excel_handler = None  # 将在首次使用时初始化
        self.logger = logging.getLogger(__name__)
        self.config = {}  # 用于存储配置信息

    def _generate_scale_values(self) -> List[float]:
        """
        生成标准 AHP 标度值列表

        Returns:
            List[float]: 包含 1/9 到 9 的标度值列表
        """
        integer_scales = list(range(1, 10))  # 1-9
        reciprocals = [1 / x for x in integer_scales[1:]]  # 1/2-1/9
        return sorted(reciprocals + integer_scales)

    def _get_excel_handler(self):
        """
        获取 Excel 处理器实例，如果未初始化则创建

        Returns:
            ExcelDataHandler: Excel 数据处理器实例
        """
        if self.excel_handler is None:
            # 动态导入 ExcelDataHandler，以避免循环导入
            from excel_handler import ExcelDataHandler
            self.excel_handler = ExcelDataHandler(self.matrix_validator)
        return self.excel_handler

    def process_level(self, level: str) -> Dict:
        """
        处理单个层次的 AHP 数据

        Args:
            level (str): 层级名称（如 'Goal', '技术风险C1'）

        Returns:
            Dict: AHP 处理结果，包括权重和一致性信息
        """
        try:
            # 构建 Excel 文件路径
            excel_path = f"{self.excel_prefix}{level.replace(' ', '')}.xlsx"

            # 读取专家判断矩阵
            matrices, criteria_names = self._get_excel_handler().read_expert_matrices(excel_path)

            # 处理专家判断
            result = self.process_expert_judgments(
                matrices,
                criteria_names,
                correction_method=self.config.get('correction_method', 'LLSM'),
                aggregation_method=self.config.get('aggregation_method', 'geometric'),
                weight_method=self.config.get('weight_method', 'eigenvector')
            )

            adjusted_matrices_count = sum(1 for corr in result["correction_results"] if corr.adjusted)

            self.logger.info(f"成功处理层级 '{level}'，共 {len(matrices)} 个专家判断，"
                             f"{adjusted_matrices_count} 个矩阵被修正 "
                             f"({sum(1 for r in result['correction_results'] if r.success)} 个达到一致性标准)")
            return result

        except Exception as e:
            error_msg = f"处理层级 '{level}' 时出错: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def process_expert_judgments(self,
                                 matrices: List[np.ndarray],
                                 criteria_names: Optional[List[str]] = None,
                                 correction_method: str = "LLSM",
                                 aggregation_method: str = "geometric",
                                 weight_method: str = "eigenvector") -> Dict:
        """
        处理专家判断矩阵，计算权重和一致性

        Args:
            matrices (List[np.ndarray]): 专家判断矩阵列表
            criteria_names (Optional[List[str]]): 准则名称列表
            correction_method (str): 矩阵修正方法
            aggregation_method (str): 矩阵聚合方法
            weight_method (str): 权重计算方法

        Returns:
            Dict: 处理结果字典，包含修正结果、权重、一致性等信息

        Raises:
            ValueError: 如果输入矩阵列表无效
        """
        # 验证输入
        if not matrices or not all(isinstance(m, np.ndarray) for m in matrices):
            error_msg = "输入矩阵列表无效"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # 获取矩阵维度
        n = matrices[0].shape[0]

        # 如果未提供准则名称，使用默认名称
        if not criteria_names:
            criteria_names = [f"准则 {i + 1}" for i in range(n)]

        # 初始化结果列表
        original_matrices = []
        corrected_matrices = []
        correction_results = []

        # 逐个处理专家判断矩阵
        for i, matrix in enumerate(matrices):
            # 检查原始矩阵一致性
            original_CR = self.consistency_checker.check_consistency(matrix).consistency_ratio

            # 矩阵是否需要修正
            needs_correction = original_CR > 0.1

            # 修正矩阵
            try:

                if needs_correction:
                    corrected_matrix, is_success, final_CR, is_adjusted = self.matrix_corrector.correct_matrix(
                        matrix, method=correction_method
                    )
                else:
                    # 如果原始矩阵已经一致，无需修正
                    corrected_matrix, is_success, final_CR, is_adjusted = matrix, True, original_CR, False
            except Exception as e:
                self.logger.error(f"修正专家 {i + 1} 的判断矩阵时出错: {str(e)}")
                # 如果修正失败，使用原始矩阵
                corrected_matrix, is_success, final_CR, is_adjusted = matrix, original_CR <= 0.1, original_CR, False

            # 记录结果
            original_matrices.append(matrix)
            corrected_matrices.append(corrected_matrix)
            correction_results.append(
                MatrixCorrectionResult(
                    expert_id=i + 1,
                    original_cr=round(original_CR, 4),
                    final_cr=round(final_CR, 4),
                    success=is_success,
                    adjusted=is_adjusted,
                    correction_method=correction_method
                )
            )

        # 聚合矩阵
        try:
            aggregated_matrix = self.matrix_aggregator.aggregate_judgments(
                corrected_matrices, method=aggregation_method
            )
        except Exception as e:
            self.logger.error(f"聚合判断矩阵时出错: {str(e)}")
            # 如果聚合失败，使用简单平均
            aggregated_matrix = np.mean(np.array(corrected_matrices), axis=0)

        # 计算最终权重
        try:
            final_weights = self.weight_calculator.calculate_weights(
                aggregated_matrix, method=weight_method
            )
        except Exception as e:
            self.logger.error(f"计算权重时出错: {str(e)}")
            # 如果计算失败，使用均等权重
            final_weights = np.ones(n) / n

        # 检查聚合矩阵一致性
        aggregated_consistency_result = self.consistency_checker.check_consistency(aggregated_matrix)

        # 构建结果字典
        return {
            "correction_results": correction_results,
            "original_matrices": original_matrices,
            "corrected_matrices": corrected_matrices,
            "aggregated_matrix": aggregated_matrix,
            "final_weights": dict(zip(criteria_names, final_weights)),
            "aggregated_consistency": aggregated_consistency_result
        }