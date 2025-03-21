# -*- coding: utf-8 -*-
import copy
import functools
import logging
import os
import traceback
from typing import List, Dict, Tuple, Optional, Callable, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


def visualization_error_handler(func):
    """
    可视化函数异常处理装饰器
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"可视化函数 {func.__name__} 执行出错: {str(e)}", exc_info=True)
            # 确保关闭任何打开的图形
            plt.close('all')
            # 如果提供了输出目录，生成错误报告
            output_dir = kwargs.get('output_dir')
            if output_dir:
                error_file = os.path.join(output_dir, f"visualization_error_{func.__name__}.txt")
                try:
                    with open(error_file, 'w', encoding='utf-8') as f:
                        f.write(f"可视化函数 {func.__name__} 执行出错: {str(e)}\n")
                        f.write(f"参数: {args}, {kwargs}\n")
                        f.write(f"异常跟踪:\n{traceback.format_exc()}")
                except:
                    pass
            return None

    return wrapper


class EnhancedFuzzyEvaluator:
    """
    增强型模糊综合评价器，支持动态隶属度函数和敏感性分析
    
    特点：
    1. 支持静态和动态隶属度函数生成
    2. 兼容敏感性分析所需的接口
    3. 提供详细的中间计算结果和可视化功能
    """

    def __init__(self,
                 risk_levels: Optional[List[str]] = None,
                 static_membership_params: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
                 dynamic_enabled: bool = True):
        """
        初始化增强型模糊评价器
        
        Args:
            risk_levels (Optional[List[str]]): 风险等级列表，默认 ["VL", "L", "M", "H", "VH"]
            static_membership_params (Optional[Dict[str, Tuple]]): 静态隶属度函数参数，默认为梯形函数
            dynamic_enabled (bool): 是否默认启用动态隶属度函数
        """
        self.risk_levels = risk_levels or ["VL", "L", "M", "H", "VH"]
        self.static_membership_params = static_membership_params or {
            "VL": (0.0, 0.1, 0.2, 0.3),
            "L": (0.2, 0.3, 0.4, 0.5),
            "M": (0.4, 0.5, 0.6, 0.7),
            "H": (0.6, 0.7, 0.8, 0.9),
            "VH": (0.8, 0.9, 1.0, 1.0)
        }
        self.dynamic_enabled = dynamic_enabled

        # 静态隶属度函数（初始化时创建）
        self.static_membership_functions = self._create_membership_functions(self.static_membership_params)

        # 动态隶属度函数（首次使用时创建）
        self.dynamic_membership_functions_cache = {}

        # 记录最近一次评价使用的隶属度函数
        self.last_used_membership_functions = None

        # 评价结果缓存
        self.evaluation_cache = {}

        # 配置日志记录器
        self.logger = logging.getLogger(__name__)

        # 初始化字体属性
        self.font_properties = self._get_chinese_font()

    def _create_membership_functions(self, params: Dict[str, Tuple[float, float, float, float]]) -> Dict[str, Callable]:
        """
        创建梯形隶属度函数
        
        Args:
            params (Dict[str, Tuple]): 每个风险等级的梯形参数 (a, b, c, d)
                a, b: 隶属度为1的左边界点
                c, d: 隶属度为1的右边界点
                
        Returns:
            Dict[str, Callable]: 隶属度函数字典 {风险等级: 隶属度函数}
        """

        def create_trap_mf(a: float, b: float, c: float, d: float) -> Callable:
            """创建单个梯形隶属度函数"""

            def trap_mf(x: float) -> float:
                """梯形隶属度函数实现"""
                if x <= a or x >= d:
                    return 0.0
                elif a < x <= b:
                    return (x - a) / (b - a + 1e-10)  # 防止除零
                elif b < x <= c:
                    return 1.0
                else:  # c < x < d
                    return (d - x) / (d - c + 1e-10)  # 防止除零

            return trap_mf

        # 创建并返回隶属度函数字典
        return {level: create_trap_mf(*param) for level, param in params.items()}

    def generate_dynamic_membership_functions(self, expert_scores: np.ndarray) -> Dict[str, callable]:
        """
        基于专家评分分布特征动态生成优化的隶属度函数

        参数:
            expert_scores: 专家评分数组(归一化后，0-1范围)

        返回:
            动态生成的隶属度函数字典
        """
        # 防止输入数据问题
        if len(expert_scores) < 4:
            logging.warning("专家评分样本量过小，使用静态隶属度函数")
            return self.static_membership_functions

        # 基础统计特征计算
        min_score = np.min(expert_scores)
        max_score = np.max(expert_scores)
        mean_score = np.mean(expert_scores)
        std_score = max(np.std(expert_scores), 0.05)  # 确保标准差不会太小

        # 分位数计算 - 扩展至更多分位点以更好刻画分布
        percentiles = [10, 25, 50, 75, 90]
        p_values = [np.percentile(expert_scores, p) for p in percentiles]
        p10, q1, q2, q3, p90 = p_values

        # 执行分布形态检测
        skewness = ((mean_score - q2) / std_score) * 3  # 分布偏度估计

        # 基于偏度调整隶属度函数参数计算
        skew_factor = max(-0.5, min(0.5, skewness * 0.25))  # 控制偏度影响范围

        # 初始化参数集
        vl_upper = min(q1, p10 + 2 * std_score)
        l_upper = q2 - skew_factor * (q2 - q1)
        m_lower = max(q1, q2 - std_score)
        m_upper = min(q3, q2 + std_score)
        h_lower = q2 + skew_factor * (q3 - q2)
        h_upper = max(q3, p90 - 2 * std_score)

        # 动态构建隶属度函数参数（优化版）
        dynamic_params = {
            "VL": (0, 0, min_score + 0.5 * std_score, vl_upper),
            "L": (min_score + 0.3 * std_score, q1, q1 + 0.5 * (q2 - q1), l_upper),
            "M": (m_lower, q2 - 0.2 * std_score, q2 + 0.2 * std_score, m_upper),
            "H": (h_lower, q3 - 0.5 * (q3 - q2), q3, h_upper),
            "VH": (max(q3, max_score - 2 * std_score), max_score - 0.5 * std_score, 1.0, 1.0)
        }

        # 参数合理性检查与调整
        params_list = [(k, v) for k, v in dynamic_params.items()]
        for i in range(len(params_list)):
            level, (a, b, c, d) = params_list[i]
            # 确保参数在[0,1]范围内
            a = max(0, min(a, 1))
            b = max(a, min(b, 1))
            c = max(b, min(c, 1))
            d = max(c, min(d, 1))
            dynamic_params[level] = (a, b, c, d)

        # 参数边界一致性检查
        for i in range(len(params_list) - 1):
            current_level, (_, _, _, d_current) = params_list[i]
            next_level, (a_next, _, _, _) = params_list[i + 1]

            # 如果边界不一致，进行调整
            if d_current < a_next:
                # 取中点作为共同边界
                common_boundary = (d_current + a_next) / 2
                dynamic_params[current_level] = dynamic_params[current_level][:3] + (common_boundary,)
                dynamic_params[next_level] = (common_boundary,) + dynamic_params[next_level][1:]
            elif d_current > a_next:
                # 确保边界有序
                common_boundary = (d_current + a_next) / 2
                dynamic_params[current_level] = dynamic_params[current_level][:3] + (common_boundary,)
                dynamic_params[next_level] = (common_boundary,) + dynamic_params[next_level][1:]

        # 记录参数用于诊断
        for level, params in dynamic_params.items():
            logging.debug(
                f"动态隶属度函数参数-{level}: a={params[0]:.3f}, b={params[1]:.3f}, c={params[2]:.3f}, d={params[3]:.3f}")

        # 创建隶属度函数
        return self._create_membership_functions(dynamic_params)

    def calculate_membership_degree(self,
                                    expert_scores: np.ndarray,
                                    expert_weights: Optional[list[float]] = None,
                                    use_dynamic: Optional[bool] = None) -> np.ndarray:
        """
        计算风险因素的隶属度向量
        
        Args:
            expert_scores (np.ndarray): 专家评分数组(归一化后，0-1范围)
            expert_weights (list[float]): FCE的专家权重
            use_dynamic (Optional[bool]): 是否使用动态隶属度函数，None表示使用默认设置
            
        Returns:
            np.ndarray: 隶属度向量 [对VL的隶属度, 对L的隶属度, ...]
        """
        # 确定是否使用动态隶属度函数
        use_dynamic = self.dynamic_enabled if use_dynamic is None else use_dynamic
        if expert_weights is None:
            logging.warning("FCE的专家权重为空，初始化为默认值")
            expert_weights = [1.0 / expert_scores.shape[0]] * expert_scores.shape[0]

        # 将专家评分转换为numpy数组并确保在[0,1]范围内
        scores = np.asarray(expert_scores, dtype=float)
        scores = np.clip(scores, 0, 1)

        # 选择或生成隶属度函数
        if use_dynamic:
            # 检查是否已有相同数据特征的缓存
            cache_key = hash(scores.tobytes())
            if cache_key in self.dynamic_membership_functions_cache:
                membership_functions = self.dynamic_membership_functions_cache[cache_key]
                self.logger.debug("使用缓存的动态隶属度函数")
            else:
                # 根据当前评分特征生成动态隶属度函数
                membership_functions = self.generate_dynamic_membership_functions(scores)
                # 缓存生成的函数（限制缓存大小）
                if len(self.dynamic_membership_functions_cache) > 100:
                    self.dynamic_membership_functions_cache.clear()
                self.dynamic_membership_functions_cache[cache_key] = membership_functions
                self.logger.debug("生成新的动态隶属度函数")
        else:
            # 使用静态隶属度函数
            membership_functions = self.static_membership_functions
            self.logger.debug("使用静态隶属度函数")

        # 记录最近使用的隶属度函数
        self.last_used_membership_functions = membership_functions

        # 计算隶属度
        membership = np.zeros(len(self.risk_levels))
        for i, level in enumerate(self.risk_levels):
            mf = membership_functions[level]
            # Calculate individual membership values for each expert score
            individual_memberships = np.array([mf(score) for score in scores])
            # Apply expert weights to individual membership values
            membership[i] = np.sum(individual_memberships * np.array(expert_weights, dtype=float))

        # 归一化处理
        sum_membership = np.sum(membership)
        if sum_membership > 0:
            membership = membership / sum_membership

        return membership

    def fuzzy_multiply(self, weight_vector: np.ndarray, membership_matrix: np.ndarray) -> np.ndarray:
        """
        加权模糊运算，计算综合隶属度
        
        Args:
            weight_vector (np.ndarray): 权重向量 [w1, w2, ...]
            membership_matrix (np.ndarray): 隶属度矩阵 [[m11, m12, ...], [m21, m22, ...], ...]
                每行对应一个风险因素，每列对应一个风险等级
                
        Returns:
            np.ndarray: 综合隶属度向量
        """
        # 确保权重向量和隶属度矩阵兼容
        if weight_vector.shape[0] != membership_matrix.shape[0]:
            raise ValueError(
                f"权重向量维度({weight_vector.shape[0]})与隶属度矩阵行数({membership_matrix.shape[0]})不匹配")

        # 确保权重归一化
        weight_sum = np.sum(weight_vector)
        if not np.isclose(weight_sum, 1.0) and weight_sum > 0:
            weight_vector = weight_vector / weight_sum
            self.logger.debug("权重向量已自动归一化")

        # 执行加权矩阵乘法 R = W * M
        result = np.dot(weight_vector, membership_matrix)

        # 归一化处理
        result_sum = np.sum(result)
        if result_sum > 0:
            result = result / result_sum

        return result

    def calculate_risk_index(self, fuzzy_vector: np.ndarray) -> float:
        """
        计算风险指数(加权平均法)
        
        Args:
            fuzzy_vector (np.ndarray): 模糊评价结果向量
            
        Returns:
            float: 风险指数值(0-1)
        """
        # 风险等级对应的数值评分
        # VL=1.5, L=3.5, M=5.5, H=7.5, VH=9.5 评分区间中位数
        level_values = np.array([1.5, 3.5, 5.5, 7.5, 9.5])

        # 计算加权平均
        risk_index = np.sum(fuzzy_vector * level_values)

        return risk_index

    def evaluate(self,
                 expert_scores: Dict[str, np.ndarray],
                 factor_weights: Dict[str, float],
                 expert_weights: Optional[list[float]] = None,
                 use_dynamic: Optional[bool] = None) -> Dict[str, Any]:
        """
        单层模糊综合评价
        
        Args:
            expert_scores (Dict[str, np.ndarray]): 专家评分 {因素: 评分数组}
            factor_weights (Dict[str, float]): 因素权重 {因素: 权重}
            expert_weights (list[float]): FCE的专家权重
            use_dynamic (Optional[bool]): 是否使用动态隶属度函数
            
        Returns:
            Dict[str, Any]: 评价结果
                {
                    "membership_matrix": 隶属度矩阵,
                    "weight_vector": 权重向量,
                    "factor_membership": {因素: 隶属度向量},
                    "integrated_result": 综合评价结果向量,
                    "risk_index": 风险指数
                }
        """
        # 提取共同的因素集合
        common_factors = set(expert_scores.keys()) & set(factor_weights.keys())
        if not common_factors:
            raise ValueError("专家评分和因素权重中没有共同的因素")

        # 转换为有序列表，确保权重向量和隶属度矩阵对应
        ordered_factors = sorted(list(common_factors))

        # 计算每个因素的隶属度
        factor_membership = {}
        membership_matrix = np.zeros((len(ordered_factors), len(self.risk_levels)))
        weight_vector = np.zeros(len(ordered_factors))

        for i, factor in enumerate(ordered_factors):
            # 计算隶属度
            normalized_scores = np.asarray(expert_scores[factor]) / 10.0  # 假设原始评分为1-10
            membership = self.calculate_membership_degree(normalized_scores, expert_weights, use_dynamic)
            factor_membership[factor] = membership
            membership_matrix[i] = membership

        # 构建权重向量
        weight_vector = np.array([factor_weights.get(factor, 0) for factor in ordered_factors])
        weight_sum = np.sum(weight_vector)
        if weight_sum > 0:
            weight_vector = weight_vector / weight_sum

        # 计算综合评价结果
        integrated_result = self.fuzzy_multiply(weight_vector, membership_matrix)

        # 计算风险指数
        risk_index = self.calculate_risk_index(integrated_result)

        # 打印调试信息（实际使用时可以移除）
        logging.debug(f"权重矢量: {weight_vector}")
        logging.debug(f"风险指数: {risk_index}")

        # 构建结果字典
        result = {
            "membership_matrix": membership_matrix,
            "weight_vector": weight_vector,
            "factor_membership": factor_membership,
            "integrated_result": integrated_result,
            "risk_index": risk_index,
            "ordered_factors": ordered_factors  # 保存因素顺序，便于后续处理
        }

        # 缓存结果
        self.evaluation_cache["latest"] = result

        return result

    def validate_sensitivity_inputs(factor_weights, expert_scores):
        """
        验证敏感性分析的输入数据有效性

        Args:
            factor_weights (Dict[str, float]): 因素权重
            expert_scores (Dict[str, np.ndarray]): 专家评分

        Returns:
            Dict[str, float]: 验证后的因素权重
        """
        validated_weights = {}

        # 验证权重值
        for factor, weight in factor_weights.items():
            # 检查权重是否为有限值且非负
            if not np.isfinite(weight) or weight < 0:
                logging.warning(f"因素 '{factor}' 的权重值无效: {weight}，设置为0")
                validated_weights[factor] = 0.0
            else:
                validated_weights[factor] = weight

        # 确保权重总和为1（如果有非零权重）
        weight_sum = sum(validated_weights.values())
        if weight_sum > 0:
            for factor in validated_weights:
                validated_weights[factor] /= weight_sum

        # 验证专家评分
        for factor, scores in expert_scores.items():
            if factor in validated_weights:
                # 检查并替换非有限值
                invalid_indices = ~np.isfinite(scores)
                if np.any(invalid_indices):
                    logging.warning(f"因素 '{factor}' 的评分包含 {np.sum(invalid_indices)} 个无效值")
                    # 使用有效评分的均值替换无效值
                    valid_scores = scores[~invalid_indices]
                    if len(valid_scores) > 0:
                        scores[invalid_indices] = np.mean(valid_scores)
                    else:
                        scores[invalid_indices] = 0.5  # 默认中间值
                    expert_scores[factor] = scores

        return validated_weights

    def normalize_sensitivity_results(self, sensitivity_indices: Dict[str, float]) -> Dict[str, float]:
        """
        规范化敏感性指标，确保结果中没有NaN或非有限值

        Args:
            sensitivity_indices (Dict[str, float]): 原始敏感性指标

        Returns:
            Dict[str, float]: 规范化后的敏感性指标
        """
        # 创建新字典以存储规范化结果
        normalized_indices = {}

        # 检查并替换非有限值
        for factor, value in sensitivity_indices.items():
            if not np.isfinite(value):
                self.logger.warning(f"因素 '{factor}' 的敏感性指标为非有限值，设为0")
                normalized_indices[factor] = 0.0
            else:
                normalized_indices[factor] = value

        # 检查是否存在有效敏感性指标
        valid_values = [v for v in normalized_indices.values() if v != 0.0]
        if not valid_values:
            self.logger.warning("所有敏感性指标均为0或无效，无法进行有效比较")
            return normalized_indices

        # 可选：对极端值进行裁剪处理
        # 计算有效值的四分位数
        q1 = np.percentile([abs(v) for v in valid_values], 25)
        q3 = np.percentile([abs(v) for v in valid_values], 75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr

        # 裁剪极端异常值
        for factor, value in normalized_indices.items():
            if abs(value) > upper_bound:
                self.logger.debug(f"因素 '{factor}' 的敏感性指标 {value} 超出上界 {upper_bound}，将被裁剪")
                normalized_indices[factor] = upper_bound if value > 0 else -upper_bound

        return normalized_indices

    @staticmethod
    def normalize_sensitivity_weights(
            factor_weights: Dict[str, float],
            modified_factor: str,
            new_weight: float,
            epsilon: float = 1e-10
    ) -> Dict[str, float]:
        """
        计算单因素扰动后的归一化权重分布 (独立工具函数)

        参数:
            factor_weights: 原始风险因素权重字典
            modified_factor: 被修改的因素标识符
            new_weight: 修改后的权重值
            epsilon: 数值稳定性阈值

        返回:
            Dict[str, float]: 保持比例关系的归一化权重分布
        """
        # 创建权重的字典副本
        if isinstance(factor_weights, dict):
            modified_weights = dict(factor_weights)
        else:
            # 处理非字典输入，尝试转换为字典
            try:
                modified_weights = dict(factor_weights)
            except (TypeError, ValueError):
                print(f"警告: 无法将输入转换为字典，返回原始输入")
                return factor_weights

        # 验证输入参数
        if modified_factor not in modified_weights:
            print(f"警告: 修改的因素'{modified_factor}'未在权重字典中找到")
            return modified_weights

        # 应用权重修改，确保数值稳定性
        modified_weights[modified_factor] = max(epsilon, new_weight)

        # 计算权重调整因子
        original_sum = sum(modified_weights.values())
        if original_sum <= epsilon:
            # 处理数值退化情况
            n_factors = len(modified_weights)
            return {k: 1.0 / n_factors for k in modified_weights}

        # 计算比例归一化的调整因子
        non_modified_sum = sum(v for k, v in modified_weights.items() if k != modified_factor)
        target_remainder = max(0.0, 1.0 - modified_weights[modified_factor])

        # 调整非修改因素的权重
        if non_modified_sum > epsilon and target_remainder > epsilon:
            # 标准比例调整
            scale = target_remainder / non_modified_sum
            for k in modified_weights:
                if k != modified_factor:
                    modified_weights[k] *= scale
        else:
            # 边缘情况：平均分配剩余权重
            n_other = len(modified_weights) - 1
            if n_other > 0 and target_remainder > epsilon:
                for k in modified_weights:
                    if k != modified_factor:
                        modified_weights[k] = target_remainder / n_other

        # 最终归一化
        weight_sum = sum(modified_weights.values())
        if weight_sum > epsilon:
            return {k: v / weight_sum for k, v in modified_weights.items()}
        else:
            # 极端情况处理
            return {k: 1.0 / len(modified_weights) for k in modified_weights}

    def calculate_normalized_weights(
            factor_weights: Dict[str, float],
            modified_factor: str,
            new_weight: float,
            epsilon: float = 1e-10
    ) -> Dict[str, float]:
        """
        Calculates normalized weight distribution with single factor perturbation

        Parameters:
            factor_weights (Dict[str, float]): Original risk factor weight dictionary
            modified_factor (str): Identifier of the factor being modified
            new_weight (float): New weight value for the modified factor
            epsilon (float): Numerical stability threshold

        Returns:
            Dict[str, float]: Normalized weight distribution preserving proportional relationships
        """
        # Create working copy of weights to prevent side effects
        modified_weights = factor_weights.copy()

        # Validate input parameters
        if modified_factor not in modified_weights:
            raise ValueError(f"Modified factor '{modified_factor}' not found in weight dictionary")

        # Apply weight modification with numerical stability constraints
        modified_weights[modified_factor] = max(epsilon, new_weight)

        # Calculate weight adjustment factor
        original_sum = sum(modified_weights.values())
        if original_sum <= epsilon:
            # Handle numerical degenerate case
            n_factors = len(modified_weights)
            return {k: 1.0 / n_factors for k in modified_weights}

        # Calculate adjustment factor for proportional normalization
        non_modified_weight_sum = original_sum - modified_weights[modified_factor]
        target_remainder = 1.0 - modified_weights[modified_factor]

        # Handle edge case where modified weight exceeds 1.0
        if target_remainder < epsilon:
            # Modified weight dominates; set all others to minimum value
            for factor in modified_weights:
                if factor != modified_factor:
                    modified_weights[factor] = epsilon
            # Adjust modified factor to ensure proper normalization
            denominator = epsilon * (len(modified_weights) - 1) + modified_weights[modified_factor]
            scaling_factor = 1.0 / denominator
        else:
            # Standard case: proportional adjustment of non-modified weights
            if non_modified_weight_sum < epsilon:
                # Handle case where other weights are effectively zero
                equal_weight = target_remainder / (len(modified_weights) - 1)
                for factor in modified_weights:
                    if factor != modified_factor:
                        modified_weights[factor] = equal_weight
            else:
                # Normal proportional adjustment
                scaling_factor = target_remainder / non_modified_weight_sum
                for factor in modified_weights:
                    if factor != modified_factor:
                        modified_weights[factor] = modified_weights[factor] * scaling_factor

        # Perform final normalization to ensure sum equals 1.0
        final_sum = sum(modified_weights.values())
        normalized_weights = {k: v / final_sum for k, v in modified_weights.items()}

        return normalized_weights

    def create_modified_weights(self, weights, modified_factor, variation):
        """创建修改后的权重分布，保持其他因素的相对比例"""
        original_weight = weights[modified_factor]
        new_weight = max(0.001, original_weight * (1 + variation))

        # 获取除修改因素外的总权重
        other_weights_sum = sum(w for f, w in weights.items() if f != modified_factor)

        # 计算其他权重需要的缩放比例
        if other_weights_sum > 0:
            scale_factor = (1 - new_weight) / other_weights_sum
        else:
            scale_factor = 0

        # 创建新的权重字典
        modified_weights = {}
        for factor, weight in weights.items():
            if factor == modified_factor:
                modified_weights[factor] = new_weight
            else:
                modified_weights[factor] = weight * scale_factor

        return modified_weights

    def calculate_sensitivity(self, factor_weights, expert_scores, factor, variation_range=0.2):
        """改进的敏感性计算方法"""
        baseline_result = self.evaluate(expert_scores, factor_weights)
        baseline_index = baseline_result["risk_index"]

        # 避免基准值过小
        if baseline_index < 1e-6:
            return 0.0

        # 使用多点拟合而不是只用两个端点
        variations = np.linspace(-variation_range, variation_range, 5)
        indices = []

        for var in variations:
            # 创建新的权重分布
            modified_weights = self.create_modified_weights(factor_weights, factor, var)
            result = self.evaluate(expert_scores, modified_weights)
            indices.append(result["risk_index"])

        # 使用线性回归计算斜率
        if len(set(indices)) <= 1:  # 如果所有风险指数相同
            return 0.0

        try:
            slope = np.polyfit(variations, indices, 1)[0]
            sensitivity = slope * factor_weights[factor] / baseline_index
            return sensitivity
        except:
            return 0.0

    def enhanced_sensitivity_analysis(self, factor_weights, expert_scores, expert_weights, use_dynamic=None):
        """重构的敏感性分析函数"""
        # 计算基准评估
        baseline_result = self.evaluate(expert_scores, factor_weights, expert_weights, use_dynamic)
        baseline_risk_index = baseline_result["risk_index"]

        # 如果基准风险指数无效，提前返回
        if not np.isfinite(baseline_risk_index) or baseline_risk_index == 0:
            logging.error("基准风险指数无效或为零，无法进行敏感性分析")
            return {"sensitivity_indices": {factor: 0.0 for factor in factor_weights}}

        # 初始化结果
        sensitivity_indices = {}
        variation_curves = {}

        # 对每个因素进行敏感性分析
        for factor in factor_weights:
            original_weight = factor_weights[factor]
            if original_weight <= 0:
                sensitivity_indices[factor] = 0.0
                continue

            delta_indices = []
            variations = [-0.2, -0.1, 0, 0.1, 0.2]  # 使用更多点以获取更稳定的估计
            indices = []

            for variation in variations:
                # 创建新的权重分布（确保正确调整其他权重）
                modified_weights = factor_weights.copy()

                # 计算调整后的权重值
                new_weight = max(0.001, original_weight * (1 + variation))
                modified_weights[factor] = new_weight

                # 重新分配其他权重，保持它们之间的相对比例
                other_sum = sum(w for f, w in factor_weights.items() if f != factor)
                remaining = 1.0 - new_weight

                if other_sum > 0:
                    # 按原比例分配剩余权重
                    ratio = remaining / other_sum
                    for f in modified_weights:
                        if f != factor:
                            modified_weights[f] = factor_weights[f] * ratio

                # 确保权重和为1
                weight_sum = sum(modified_weights.values())
                if abs(weight_sum - 1.0) > 1e-6:
                    modified_weights = {k: v / weight_sum for k, v in modified_weights.items()}

                # 打印调试信息
                print(f"Factor: {factor}, Variation: {variation}, Weights: {modified_weights}")

                # 重要：使用新权重进行完整评估
                result = self.evaluate(expert_scores, modified_weights, expert_weights, use_dynamic)
                risk_index = result["risk_index"]

                # 打印调试信息
                print(f"  -> Risk Index: {risk_index}")

                indices.append(risk_index)

                # 计算相对变化
                if variation != 0:
                    delta = (risk_index - baseline_risk_index) / (variation * original_weight)
                    delta_indices.append(delta)

            # 如果有有效的变化
            if delta_indices:
                # 使用平均相对变化作为敏感性指标
                sensitivity = np.mean(delta_indices)
                if not np.isfinite(sensitivity):
                    sensitivity = 0.0
            else:
                sensitivity = 0.0

            sensitivity_indices[factor] = sensitivity
            variation_curves[factor] = {"variations": variations, "risk_indices": indices}

        # 添加以下调试代码
        print("=============================================")
        print("原始权重:", factor_weights)
        print("基准风险指数:", baseline_risk_index)
        print("基准模糊评价结果:", baseline_result["integrated_result"])

        for factor in factor_weights:
            print(f"\n分析因素: {factor}, 原始权重: {factor_weights[factor]}")

            # 重要：验证权重修改是否生效
            test_weights = factor_weights.copy()
            test_weights[factor] *= 1.2  # 增加20%

            # 确保权重和为1
            weight_sum = sum(test_weights.values())
            test_weights = {k: v / weight_sum for k, v in test_weights.items()}

            print("测试修改权重:", test_weights)

            # 重新评估
            test_result = self.evaluate(expert_scores, test_weights, expert_weights, use_dynamic)
            print("修改后风险指数:", test_result["risk_index"])
            print("修改后模糊评价结果:", test_result["integrated_result"])

            # 计算简单敏感性
            weight_change = (test_weights[factor] - factor_weights[factor]) / factor_weights[factor]
            index_change = (test_result["risk_index"] - baseline_risk_index) / baseline_risk_index

            if weight_change != 0:
                simple_sensitivity = index_change / weight_change
                print(f"简单敏感性估计: {simple_sensitivity}")
            else:
                print("权重未发生实际变化")

        return {
            "sensitivity_indices": sensitivity_indices,
            "variation_curves": variation_curves,
            "baseline_risk_index": baseline_risk_index
        }

    def enhanced_sensitivity_analysis_(self, factor_weights: Dict[str, float],
                                       expert_scores: Dict[str, np.ndarray],
                                       use_dynamic: bool = None,
                                       variation_range: float = 0.2,
                                       steps: int = 10) -> Dict[str, Any]:
        """
        Enhanced sensitivity analysis with robust calculation methodology
        """
        # Calculate baseline risk index
        baseline_result = self.evaluate(expert_scores, factor_weights, use_dynamic)
        baseline_risk_index = baseline_result["risk_index"]
        ordered_factors = list(factor_weights.keys())

        # Initialize result containers
        sensitivity_indices = {}
        variation_curves = {}

        # Apply log transformation to prevent numerical underflow
        epsilon = 1e-10  # Small constant to prevent logarithm of zero

        # Create variation range once - fix for the undefined variable
        variation_percentages = np.linspace(-variation_range, variation_range, steps + 1)

        # For each factor, calculate sensitivity with robust numerical approach
        for factor in ordered_factors:
            risk_indices = []
            original_weight = factor_weights[factor]

            # Skip factors with zero weight
            if original_weight < epsilon:
                sensitivity_indices[factor] = 0.0
                continue

            # Iterate through variations
            for variation in variation_percentages:
                # Calculate new weight with bounds checking
                new_weight = max(epsilon, original_weight * (1 + variation))

                # Create modified weights with normalization
                modified_weights = self.normalize_sensitivity_weights(factor_weights, factor, new_weight)

                # Calculate risk with modified weights
                result = self.evaluate(expert_scores, modified_weights, use_dynamic)
                risk_indices.append(result["risk_index"])

            # Calculate sensitivity with robust approach
            try:
                sensitivity = self.calculate_sensitivity(factor_weights, expert_scores, factor, variation_range=0.2)
                # Use central difference approximation for derivative
                # central_diff = (risk_indices[-1] - risk_indices[0]) / (2 * variation_range)

                ## Calculate elasticity (normalized sensitivity)
                # if baseline_risk_index > epsilon:
                #    sensitivity = central_diff * original_weight / baseline_risk_index
                # else:
                #    sensitivity = central_diff * original_weight / epsilon

                # Apply bounded normalization to prevent overflow
                sensitivity = np.clip(sensitivity, -1.0, 1.0)
            except Exception as e:
                print(f"计算因素 '{factor}' 敏感性时出错: {str(e)}")
                sensitivity = 0.0

            sensitivity_indices[factor] = sensitivity
            # Store both percentages and calculated indices - fixed line
            variation_curves[factor] = {
                "variations": variation_percentages.tolist(),  # Convert to list for JSON serialization
                "risk_indices": risk_indices
            }

        return {
            "sensitivity_indices": sensitivity_indices,
            "variation_curves": variation_curves,
            "ranked_factors": sorted(sensitivity_indices.keys(),
                                     key=lambda x: abs(sensitivity_indices[x]),
                                     reverse=True),
            "baseline_result": baseline_result
        }

    def perform_sensitivity_analysis(self,
                                     factor_weights: Dict[str, float],
                                     expert_scores: Dict[str, np.ndarray],
                                     expert_weights: list[float],
                                     use_dynamic: Optional[bool] = None,
                                     variation_range: float = 0.2,
                                     steps: int = 10) -> Dict[str, Any]:
        """
        单因素敏感性分析，评估权重变化对风险评估的影响

        Args:
            factor_weights (Dict[str, float]): 因素权重
            expert_scores (Dict[str, np.ndarray]): 专家评分
            expert_weights list[float]: FCE专家权重
            use_dynamic (Optional[bool]): 是否使用动态隶属度函数
            variation_range (float): 权重变化范围 (±)
            steps (int): 变化步数

        Returns:
            Dict[str, Any]: 敏感性分析结果
        """
        # 计算基准评价结果
        baseline_result = self.evaluate(expert_scores, factor_weights, expert_weights, use_dynamic)
        baseline_risk_index = baseline_result["risk_index"]
        ordered_factors = baseline_result["ordered_factors"]

        # 初始化结果容器
        sensitivity_indices = {}
        variation_curves = {}

        # 计算权重变化步长
        step_size = 2 * variation_range / steps
        variations = [-variation_range + i * step_size for i in range(steps + 1)]

        # 对每个因素进行敏感性分析
        for factor in ordered_factors:
            risk_indices = []
            original_weight = factor_weights[factor]

            # 在变化范围内改变权重
            for variation in variations:
                # 创建修改后的权重字典
                modified_weights = copy.deepcopy(factor_weights)

                # 计算新权重 (确保不为负)
                new_weight = max(0.001, original_weight * (1 + variation))
                modified_weights[factor] = new_weight

                # 归一化
                weight_sum = sum(modified_weights.values())
                modified_weights = {k: v / weight_sum for k, v in modified_weights.items()}

                # 使用修改后的权重重新计算风险评价
                result = self.evaluate(expert_scores, modified_weights, expert_weights, use_dynamic)
                risk_indices.append(result["risk_index"])

            # 计算敏感性指标 (风险指数变化率与权重变化率之比)
            try:
                # 使用线性回归斜率计算敏感性
                slope = np.polyfit(variations, risk_indices, 1)[0]
                # 验证分母是否有效
                if abs(original_weight) < 1e-6 or abs(baseline_risk_index) < 1e-6:
                    sensitivity = 0.0
                    self.logger.warning(f"因素 '{factor}' 的权重或基准风险指数接近零，敏感性设为0")
                else:
                    sensitivity = slope * original_weight / baseline_risk_index

                # 验证结果是否有效
                if not np.isfinite(sensitivity):
                    sensitivity = 0.0
                    self.logger.warning(f"因素 '{factor}' 的敏感性计算结果非有限值，设为0")
            except Exception as e:
                self.logger.warning(f"计算因素 '{factor}' 的敏感性时出错: {str(e)}")
                sensitivity = 0.0

            # 存储结果
            sensitivity_indices[factor] = sensitivity
            variation_curves[factor] = {
                "variations": variations,
                "risk_indices": risk_indices
            }

        # 应用规范化函数处理敏感性指标
        normalized_indices = self.normalize_sensitivity_results(sensitivity_indices)

        # 按敏感性排序因素
        ranked_factors = sorted(normalized_indices.items(), key=lambda x: abs(x[1]), reverse=True)
        ranked_factor_names = [item[0] for item in ranked_factors]

        # 识别关键风险因素 (敏感性指标大于平均值的因素)
        valid_sensitivities = [abs(s) for s in normalized_indices.values() if np.isfinite(s)]
        mean_sensitivity = np.mean(valid_sensitivities) if valid_sensitivities else 0.0
        critical_factors = [factor for factor, sens in ranked_factors if abs(sens) > mean_sensitivity]

        # 整合结果
        result = {
            "sensitivity_indices": normalized_indices,
            "variation_curves": variation_curves,
            "ranked_factors": ranked_factor_names,
            "baseline_result": baseline_result,
            "critical_factors": critical_factors,
            "mean_sensitivity": mean_sensitivity
        }

        # 缓存结果
        self.evaluation_cache["latest_sensitivity"] = result
        return result

    def _generate_modified_weights(self,
                                   original_weights: Dict[str, float],
                                   factor_changes: Dict[str, float],
                                   epsilon: float = 1e-10) -> Dict[str, float]:
        """
        生成修改后的权重分布，同时支持修改多个因素的权重

        Args:
            original_weights (Dict[str, float]): 原始权重字典
            factor_changes (Dict[str, float]): 要修改的因素及其变化比例 {因素名: 变化比例}
                                              例如 {'风险1': 1.2} 表示风险1的权重增加20%
            epsilon (float): 数值稳定性阈值

        Returns:
            Dict[str, float]: 修改后的权重字典，总和为1
        """
        # 创建修改后的权重字典副本
        modified_weights = original_weights.copy()

        # 计算原始的未修改因素总权重
        unmodified_factors = [f for f in original_weights if f not in factor_changes]
        unmodified_weight_sum = sum(original_weights[f] for f in unmodified_factors)

        # 应用变化比例修改指定因素的权重
        for factor, change_ratio in factor_changes.items():
            if factor in original_weights:
                # 应用变化比例（确保非负）
                modified_weights[factor] = max(epsilon, original_weights[factor] * change_ratio)

        # 计算修改后的因素总权重
        modified_factors_weight = sum(modified_weights[f] for f in factor_changes if f in modified_weights)

        # 处理特殊情况：如果总权重已超过1
        if modified_factors_weight >= 1.0:
            # 按比例缩小修改因素的权重
            scale = (1.0 - epsilon) / modified_factors_weight
            for factor in factor_changes:
                if factor in modified_weights:
                    modified_weights[factor] *= scale

            # 给未修改因素分配极小权重
            for factor in unmodified_factors:
                modified_weights[factor] = epsilon / len(unmodified_factors) if len(unmodified_factors) > 0 else 0
        else:
            # 正常情况：为未修改因素按比例分配剩余权重
            remaining_weight = 1.0 - modified_factors_weight

            if unmodified_weight_sum > epsilon:
                # 按原有比例分配
                scale = remaining_weight / unmodified_weight_sum
                for factor in unmodified_factors:
                    modified_weights[factor] = original_weights[factor] * scale
            elif len(unmodified_factors) > 0:
                # 均分剩余权重
                equal_weight = remaining_weight / len(unmodified_factors)
                for factor in unmodified_factors:
                    modified_weights[factor] = equal_weight

        # 最终归一化
        weight_sum = sum(modified_weights.values())
        if abs(weight_sum - 1.0) > epsilon and weight_sum > epsilon:
            return {k: v / weight_sum for k, v in modified_weights.items()}

        return modified_weights

    def _calculate_cross_sensitivity_matrix(self,
                                            factor1_values: np.ndarray,
                                            factor2_values: np.ndarray,
                                            baseline_risk: float,
                                            variation_steps: int) -> np.ndarray:
        """
        计算交叉敏感性矩阵，增强NaN处理能力

        Args:
            factor1_values (np.ndarray): 第一个因素的变化比例数组 (如[0.8, 0.9, 1.0, 1.1, 1.2])
            factor2_values (np.ndarray): 第二个因素的变化比例数组
            baseline_risk (float): 基准风险值
            variation_steps (int): 变化步数

        Returns:
            np.ndarray: 交叉敏感性矩阵，包含适当的NaN处理
        """
        # 验证输入数据完整性
        if not np.isfinite(baseline_risk):
            logging.error("基准风险值非有限数值")
            return np.full((variation_steps + 1, variation_steps + 1), np.nan)

        # 预处理输入数组 - 将无穷值替换为NaN
        factor1_values = np.where(np.isfinite(factor1_values), factor1_values, np.nan)
        factor2_values = np.where(np.isfinite(factor2_values), factor2_values, np.nan)

        # 计算有效数据点的百分比
        valid_f1 = np.sum(np.isfinite(factor1_values))
        valid_f2 = np.sum(np.isfinite(factor2_values))
        valid_data_ratio = (valid_f1 * valid_f2) / (len(factor1_values) * len(factor2_values))

        # 如果有效数据不足则终止
        if valid_data_ratio < 0.5:  # 可配置阈值
            logging.warning(f"交叉敏感性分析的有效数据不足 ({valid_data_ratio:.2%})")
            # 返回NaN矩阵而非None，以保持接口一致性
            return np.full((variation_steps + 1, variation_steps + 1), np.nan)

        # 生成交叉敏感性矩阵，明确处理NaN
        sensitivity_matrix = np.full((variation_steps + 1, variation_steps + 1), np.nan)

        # 计算风险指数矩阵
        for i, var1 in enumerate(factor1_values):
            for j, var2 in enumerate(factor2_values):
                # 跳过NaN输入值
                if not (np.isfinite(var1) and np.isfinite(var2)):
                    continue

                try:
                    # 创建修改后的权重分布
                    modified_weights = self._generate_modified_weights(
                        self.current_factor_weights,
                        {self.current_factors[0]: var1, self.current_factors[1]: var2}
                    )

                    # 使用修改后的权重计算风险指数
                    result = self.evaluate(
                        self.current_expert_scores,
                        modified_weights,
                        self.current_expert_weights,
                        self.dynamic_enabled
                    )

                    # 存储风险指数
                    sensitivity_matrix[i, j] = result["risk_index"]
                except Exception as e:
                    logging.error(f"计算位置 ({i},{j}) 的敏感性出错: {str(e)}")
                    # 保持为NaN

        return sensitivity_matrix

    def cross_sensitivity_analysis(self,
                                   factor_weights: Dict[str, float],
                                   expert_scores: Dict[str, np.ndarray],
                                   factors: List[str],
                                   expert_weights: Optional[list[float]] = None,
                                   use_dynamic: Optional[bool] = None,
                                   variation_range: float = 0.2,
                                   steps: int = 5) -> Dict[str, Any]:
        """
        交叉敏感性分析，评估两个风险因素同时变化的影响
        """
        if len(factors) != 2:
            raise ValueError("交叉敏感性分析需要指定两个风险因素")

        # 确保因素存在于权重和评分中
        for factor in factors:
            if factor not in factor_weights or factor not in expert_scores:
                raise ValueError(f"风险因素 '{factor}' 在权重或评分中不存在")

        # 计算基准评价结果
        # 保存当前分析状态供辅助方法使用
        self.current_factor_weights = factor_weights
        self.current_expert_scores = expert_scores
        self.current_expert_weights = expert_weights
        self.current_factors = factors
        self.dynamic_enabled = use_dynamic if use_dynamic is not None else self.dynamic_enabled

        baseline_result = self.evaluate(expert_scores, factor_weights, expert_weights, use_dynamic)
        baseline_risk_index = baseline_result["risk_index"]

        # 计算权重变化步长和变化率列表
        step_size = 2 * variation_range / steps
        variations = [-variation_range + i * step_size for i in range(steps + 1)]

        # 提取两个因素
        factor1, factor2 = factors

        # 修复：将变化率转换为变化比例（1+变化率）
        # 这是关键修复点！
        factor1_values = [1 + v for v in variations]  # 例如：[0.8, 0.9, 1.0, 1.1, 1.2]
        factor2_values = [1 + v for v in variations]  # 例如：[0.8, 0.9, 1.0, 1.1, 1.2]

        # 使用辅助方法生成敏感性矩阵，传递正确的参数类型
        risk_matrix = self._calculate_cross_sensitivity_matrix(
            factor1_values=factor1_values,  # 变化比例，而非变化率
            factor2_values=factor2_values,  # 变化比例，而非变化率
            baseline_risk=baseline_risk_index,
            variation_steps=steps
        )

        # 整合结果
        result = {
            "risk_matrix": risk_matrix,
            "variations": variations,  # 保持原始变化率用于显示
            "factors": factors,
            "baseline_risk_index": baseline_risk_index,
            "has_valid_data": not np.all(np.isnan(risk_matrix))
        }

        # 缓存结果
        self.evaluation_cache["latest_cross_sensitivity"] = result

        return result

    @staticmethod
    def detect_available_fonts():
        """
        检测系统中可用的中文字体

        返回:
            List[str]: 检测到的可能支持中文的字体列表
        """
        import matplotlib.font_manager as fm

        print("检测系统中可用的字体...")

        # 查找系统字体
        system_fonts = fm.findSystemFonts(fontpaths=None)
        available_fonts = []

        # 可能的中文字体名称片段
        chinese_font_fragments = [
            'chinese', 'china', '中文', 'cjk', 'zh',
            'ming', 'song', 'kai', 'hei', 'gothic',
            'yahei', 'simhei', 'simsun', 'msyh', 'pingfang',
            'heiti', 'micro hei', 'wenquanyi', 'wqy'
        ]

        # 检测中文字体
        for font_path in system_fonts:
            try:
                font_prop = fm.FontProperties(fname=font_path)
                font_name = font_prop.get_name()
                font_name_lower = font_name.lower()

                # 检查字体名称是否包含中文相关字符串
                if any(fragment in font_name_lower for fragment in chinese_font_fragments):
                    available_fonts.append((font_name, font_path))
            except Exception:
                continue

        print(f"检测到 {len(available_fonts)} 个可能支持中文的字体")
        for i, (name, path) in enumerate(available_fonts[:10], 1):
            print(f"{i}. {name} - {path}")

        if len(available_fonts) > 10:
            print(f"... 还有 {len(available_fonts) - 10} 个字体未显示")

        return available_fonts

    # 添加新的字体获取方法
    def _get_chinese_font(self):
        """
        获取适用于当前系统的中文本体

        Returns:
            FontProperties: 中文本体属性对象
        """
        import platform as sys_platform
        import os
        from matplotlib.font_manager import FontProperties

        system_name = sys_platform.system()

        try:
            # 基于操作系统类型实施不同的字体策略
            if system_name == 'Windows':
                # Windows系统使用微软雅黑或黑体
                font_paths = [
                    r'C:\Windows\Fonts\msyh.ttc',  # 微软雅黑
                    r'C:\Windows\Fonts\simhei.ttf',  # 黑体
                    r'C:\Windows\Fonts\simsun.ttc'  # 宋体
                ]

                for path in font_paths:
                    if os.path.exists(path):
                        font_prop = FontProperties(fname=path)
                        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
                        self.logger.debug(f"使用Windows字体: {path}")
                        return font_prop

            elif system_name == 'Darwin':  # macOS
                # macOS系统使用苹方或华文黑体
                font_paths = [
                    '/System/Library/Fonts/PingFang.ttc',
                    '/Library/Fonts/Hiragino Sans GB.ttc',
                    '/System/Library/Fonts/STHeiti Light.ttc'
                ]

                for path in font_paths:
                    if os.path.exists(path):
                        font_prop = FontProperties(fname=path)
                        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti']
                        self.logger.debug(f"使用macOS字体: {path}")
                        return font_prop

            else:  # Linux和其他系统
                # 尝试使用常见的中文本体
                import matplotlib as mpl
                chinese_fonts = [
                    'WenQuanYi Micro Hei', 'Droid Sans Fallback', 'Noto Sans CJK SC',
                    'SimSun', 'SimHei', 'WenQuanYi Zen Hei', 'AR PL UMing CN'
                ]

                # 检测系统中是否有这些字体
                available_fonts = mpl.font_manager.findSystemFonts(fontpaths=None)
                for font in available_fonts:
                    try:
                        temp_prop = FontProperties(fname=font)
                        font_name = temp_prop.get_name()
                        if any(cf.lower() in font_name.lower() for cf in chinese_fonts):
                            plt.rcParams['font.sans-serif'] = [font_name]
                            self.logger.debug(f"使用Linux字体: {font}")
                            return temp_prop
                    except Exception:
                        continue

            # 如果未找到合适字体，尝试通过字体名称设置
            for font_name in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']:
                try:
                    plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                    font_prop = FontProperties(family=font_name)
                    self.logger.debug(f"尝试使用字体名称: {font_name}")
                    return font_prop
                except Exception:
                    continue

            # 最后的回退选项
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']
            self.logger.warning("未找到合适的中文本体，使用系统默认字体，中文显示可能不正确")

        except Exception as e:
            # 记录详细的错误信息以便调试
            self.logger.error(f"配置中文本体出错: {str(e)}", exc_info=True)

        # 正确显示负号
        plt.rcParams['axes.unicode_minus'] = False

        # 返回默认字体
        return FontProperties()

    @staticmethod
    def configure_chinese_font():
        """
        配置适用于当前操作系统的中文字体

        Returns:
            Tuple[FontProperties, bool]: 中文字体属性对象和配置成功标志
        """
        # 正确导入platform模块并立即使用，避免变量覆盖
        import platform as sys_platform
        system_name = sys_platform.system()

        font_prop = None
        config_success = False

        try:
            # 基于操作系统类型实施不同的字体策略
            if system_name == 'Windows':
                # Windows系统使用微软雅黑或黑体
                font_paths = [
                    r'C:\Windows\Fonts\msyh.ttc',  # 微软雅黑
                    r'C:\Windows\Fonts\simhei.ttf',  # 黑体
                    r'C:\Windows\Fonts\simsun.ttc'  # 宋体
                ]

                for path in font_paths:
                    if os.path.exists(path):
                        font_prop = FontProperties(fname=path)
                        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
                        logging.debug(f"使用Windows字体: {path}")
                        config_success = True
                        break

            elif system_name == 'Darwin':  # macOS
                # macOS系统使用苹方或华文黑体
                font_paths = [
                    '/System/Library/Fonts/PingFang.ttc',
                    '/Library/Fonts/Hiragino Sans GB.ttc',
                    '/System/Library/Fonts/STHeiti Light.ttc'
                ]

                for path in font_paths:
                    if os.path.exists(path):
                        font_prop = FontProperties(fname=path)
                        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti']
                        logging.debug(f"使用macOS字体: {path}")
                        config_success = True
                        break

            else:  # Linux和其他系统
                # 尝试使用常见的中文字体
                import matplotlib as mpl
                chinese_fonts = [
                    'WenQuanYi Micro Hei', 'Droid Sans Fallback', 'Noto Sans CJK SC',
                    'SimSun', 'SimHei', 'WenQuanYi Zen Hei', 'AR PL UMing CN'
                ]

                # 检测系统中是否有这些字体
                available_fonts = mpl.font_manager.findSystemFonts(fontpaths=None)
                for font in available_fonts:
                    try:
                        temp_prop = FontProperties(fname=font)
                        font_name = temp_prop.get_name()
                        if any(cf.lower() in font_name.lower() for cf in chinese_fonts):
                            font_prop = FontProperties(fname=font)
                            plt.rcParams['font.sans-serif'] = [font_name]
                            logging.debug(f"使用Linux字体: {font}")
                            config_success = True
                            break
                    except Exception as e:
                        continue

            # 如果未找到合适字体，尝试通过字体名称设置
            if font_prop is None:
                # 尝试使用系统可用的字体名称
                for font_name in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']:
                    try:
                        plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                        # 尝试创建字体属性
                        font_prop = FontProperties(family=font_name)
                        logging.debug(f"尝试使用字体名称: {font_name}")
                        config_success = True
                        break
                    except:
                        continue

            # 最后的回退选项
            if font_prop is None:
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']
                logging.warning("未找到合适的中文字体，使用系统默认字体，中文显示可能不正确")
                font_prop = FontProperties()

            # 正确显示负号
            plt.rcParams['axes.unicode_minus'] = False

            return font_prop

        except Exception as e:
            # 记录详细的错误信息以便调试
            logging.error(f"配置中文字体出错: {str(e)}", exc_info=True)
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return FontProperties()

    def visualize_membership_functions(self,
                                       use_dynamic: bool = True,
                                       scores: Optional[np.ndarray] = None,
                                       output_path: Optional[str] = None):
        """
        可视化隶属度函数

        参数:
            use_dynamic: 是否使用动态隶属度函数
            scores: 用于生成动态隶属度函数的评分数据
            output_path: 输出文件路径
        """
        # 获取中文字体配置 - 直接使用实例中的字体属性
        font_prop = self.font_properties

        # 确定使用哪种隶属度函数
        if use_dynamic:
            if scores is None:
                if self.last_used_membership_functions:
                    membership_functions = self.last_used_membership_functions
                    logging.info("使用最近一次的动态隶属度函数进行可视化")
                else:
                    logging.warning("未提供评分数据且无缓存的动态隶属度函数，将使用静态函数")
                    membership_functions = self.static_membership_functions
            else:
                # 使用提供的评分生成动态隶属度函数
                normalized_scores = np.asarray(scores) / 10.0 if np.max(scores) > 1 else scores
                membership_functions = self.generate_dynamic_membership_functions(normalized_scores)
        else:
            membership_functions = self.static_membership_functions

        # 创建x轴数据点
        x = np.linspace(0, 1, 100)

        # 计算每个风险等级在各点的隶属度
        plt.figure(figsize=(10, 6))

        for level in self.risk_levels:
            mf = membership_functions[level]
            y = [mf(xi) for xi in x]
            plt.plot(x, y, label=level, linewidth=2)

        # 设置图表标题和标签（使用中文字体）
        title_str = "风险等级隶属度函数" + (" (动态)" if use_dynamic else " (静态)")
        plt.title(title_str, fontproperties=font_prop, fontsize=14)
        plt.xlabel("归一化评分", fontproperties=font_prop, fontsize=12)
        plt.ylabel("隶属度", fontproperties=font_prop, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(prop=font_prop)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def safe_bar_annotation(ax, bars, precision=4):
        """
        安全地为条形图添加数值标签，避免非有限值导致的渲染问题

        Args:
            ax: Matplotlib轴对象
            bars: 条形图对象集合
            precision: 数值精度
        """
        for bar in bars:
            height = bar.get_height()
            # 检查高度是否为有限值
            if np.isfinite(height) and height != 0:
                # 计算文本位置，确保不会超出图表边界
                text_y = height * 1.01 if height > 0 else height * 0.99

                # 添加文本标签
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    text_y,
                    f'{height:.{precision}f}',
                    ha='center',
                    va='bottom' if height >= 0 else 'top',
                    fontsize=9
                )

    @visualization_error_handler
    def visualize_sensitivity_analysis(self,
                                       sensitivity_results: Optional[Dict[str, Any]] = None,
                                       top_n: int = 5,
                                       output_dir: Optional[str] = None):
        """
        可视化敏感性分析结果

        参数:
            sensitivity_results: 敏感性分析结果
            top_n: 显示前n个最敏感的因素
            output_dir: 输出目录
        """
        # 获取中文字体配置
        font_prop = self.configure_chinese_font()

        # 获取敏感性分析结果
        if sensitivity_results is None:
            if "latest_sensitivity" in self.evaluation_cache:
                sensitivity_results = self.evaluation_cache["latest_sensitivity"]
            else:
                raise ValueError("未提供敏感性分析结果且无缓存结果可用")

        # 提取数据
        sensitivity_indices = sensitivity_results["sensitivity_indices"]
        ranked_factors = sensitivity_results["ranked_factors"]
        variation_curves = sensitivity_results["variation_curves"]

        # 如果需要保存图片，确保目录存在
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            # 1. 绘制敏感性指标条形图
            plt.figure(figsize=(12, 6))
            # 提取前top_n个因素数据
            top_factors = ranked_factors[:min(top_n, len(ranked_factors))]
            top_indices = [sensitivity_indices[f] for f in top_factors]

            # 为正负值设置不同颜色
            colors = ['#3498db' if v >= 0 else '#e74c3c' for v in top_indices]

            # 绘制条形图
            bars = plt.bar(top_factors, top_indices, color=colors)

            # 添加数值标签
            for bar, value in zip(bars, top_indices):
                if not np.isfinite(value):
                    continue

                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.,
                         0.01 if height < 0 else height + 0.01,
                         f'{value:.4f}',
                         ha='center', va='bottom' if height >= 0 else 'top',
                         fontsize=9)

            # 添加水平参考线
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            # 添加平均敏感性线 - 确保有效性并添加标签
            mean_sensitivity = sensitivity_results.get("mean_sensitivity", 0)
            if np.isfinite(mean_sensitivity):
                # 保存返回的Line2D对象以确保图例能够找到它
                avg_line = plt.axhline(y=mean_sensitivity, color='green',
                                       linestyle='--', alpha=0.7,
                                       label='平均敏感性')

                # 确保至少有一个带标签的对象用于图例
                if avg_line is not None:
                    plt.legend(prop=font_prop)

            # 使用中文字体
            plt.title("风险因素敏感性分析", fontproperties=font_prop, fontsize=14)
            plt.xlabel("风险因素", fontproperties=font_prop, fontsize=12)
            plt.ylabel("敏感性指标", fontproperties=font_prop, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if output_dir:
                plt.savefig(os.path.join(output_dir, "sensitivity_indices.png"), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        except Exception as e:
            logging.error(f"绘制敏感性指标图出错: {str(e)}")
            plt.close()

        # 2. 绘制敏感性曲线图
        try:
            plt.figure(figsize=(12, 6))

            # 跟踪是否至少有一个有效曲线
            has_valid_curves = False

            for factor in top_factors:
                if factor not in variation_curves:
                    logging.warning(f"因素 '{factor}' 在变化曲线数据中不存在，将被跳过")
                    continue

                curve = variation_curves[factor]

                # 数据有效性检查
                if "variations" not in curve or "risk_indices" not in curve:
                    logging.warning(f"因素 '{factor}' 的曲线数据结构不完整")
                    continue

                variations = curve["variations"]
                risk_indices = curve["risk_indices"]

                if len(variations) != len(risk_indices):
                    logging.warning(f"因素 '{factor}' 的变化率与风险指数数据长度不匹配")
                    continue

                # 确保数据为有限值
                valid_data = [(v, r) for v, r in zip(variations, risk_indices)
                              if np.isfinite(v) and np.isfinite(r)]

                if not valid_data:
                    logging.warning(f"因素 '{factor}' 没有有效的数据点")
                    continue

                v_values, r_values = zip(*valid_data)

                # 绘制曲线并明确设置标签
                plt.plot([v * 100 for v in v_values], r_values,
                         marker='o', linewidth=2, label=factor)

                has_valid_curves = True

            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

            # 使用中文字体
            plt.title("风险因素权重变化敏感性曲线", fontproperties=font_prop, fontsize=14)
            plt.xlabel("权重变化率 (%)", fontproperties=font_prop, fontsize=12)
            plt.ylabel("风险指数", fontproperties=font_prop, fontsize=12)
            plt.grid(True, alpha=0.3)

            # 仅当有有效曲线时添加图例
            if has_valid_curves:
                plt.legend(prop=font_prop)

            plt.tight_layout()

            if output_dir:
                plt.savefig(os.path.join(output_dir, "sensitivity_curves.png"), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        except Exception as e:
            logging.error(f"绘制敏感性曲线图出错: {str(e)}")
            plt.close()

    def visualize_cross_sensitivity(self,
                                    cross_results: Optional[Dict[str, Any]] = None,
                                    output_dir: Optional[str] = None):
        """
        可视化交叉敏感性分析结果，增强NaN处理能力

        参数:
            cross_results: 交叉敏感性分析结果
            output_dir: 输出目录
        """
        # 获取交叉敏感性分析结果
        if cross_results is None:
            if "latest_cross_sensitivity" in self.evaluation_cache:
                cross_results = self.evaluation_cache["latest_cross_sensitivity"]
            else:
                raise ValueError("未提供交叉敏感性分析结果且无缓存结果可用")

        # 提取数据
        risk_matrix = cross_results["risk_matrix"]
        variations = [f"{v * 100:.0f}%" for v in cross_results["variations"]]
        factors = cross_results["factors"]
        baseline = cross_results["baseline_risk_index"]
        has_valid_data = cross_results.get("has_valid_data", True)

        # 如果需要保存图片，确保目录存在
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 生成标题
        title = f"交叉敏感性分析: {factors[0]} vs {factors[1]}"

        # 使用增强的热图绘制方法
        output_path = os.path.join(output_dir, "cross_sensitivity.png") if output_dir else None

        from visualizer import SensitivityVisualizer
        sensitivity_visualizer = SensitivityVisualizer()
        # 如果数据有效，绘制完整热图
        if has_valid_data:
            sensitivity_visualizer.plot_heatmap(
                matrix_data=risk_matrix,
                row_labels=variations,
                col_labels=variations,
                title=title,
                output_path=output_path,
                annotate=True
            )
        else:
            # 数据无效，绘制警告信息
            logging.warning("无法生成交叉敏感性热图：数据不足")

            # 创建简单的消息图表
            plt.figure(figsize=(10, 8))
            plt.text(0.5, 0.5,
                     "无法生成交叉敏感性热图\n数据质量不足以进行有效分析",
                     ha='center', va='center',
                     fontproperties=self.font_properties,
                     fontsize=14)
            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()

    def visualize_sensitivity_uncertainty(sensitivity_results):
        """
        可视化敏感性分析的不确定性

        Args:
            sensitivity_results (Dict): 敏感性分析结果
        """
        plt.figure(figsize=(12, 6))

        factors = list(sensitivity_results["direct_sensitivity"].keys())
        direct_sensitivity = [sensitivity_results["direct_sensitivity"][f] for f in factors]
        confidence_intervals = [sensitivity_results["confidence_intervals"][f] for f in factors]

        # 绘制带置信区间的条形图
        plt.errorbar(
            factors,
            direct_sensitivity,
            yerr=[
                [ds - ci[0] for ds, ci in zip(direct_sensitivity, confidence_intervals)],
                [ci[1] - ds for ds, ci in zip(direct_sensitivity, confidence_intervals)]
            ],
            fmt='o',
            capsize=5,
            capthick=2
        )

        plt.title("风险因素敏感性分析与不确定性")
        plt.xlabel("风险因素")
        plt.ylabel("敏感性指标")
        plt.xticks(rotation=45)
        plt.tight_layout()
