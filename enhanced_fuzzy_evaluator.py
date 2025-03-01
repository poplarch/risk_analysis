# -*- coding: utf-8 -*-
import copy
import functools
import logging
import os
import traceback
from typing import List, Dict, Tuple, Optional, Callable, Any

import matplotlib
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
            raise ValueError(f"权重向量维度({weight_vector.shape[0]})与隶属度矩阵行数({membership_matrix.shape[0]})不匹配")
        
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
        # VL=0.1, L=0.3, M=0.5, H=0.7, VH=0.9
        level_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
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
    
    def perform_sensitivity_analysis(self, 
                                    factor_weights: Dict[str, float],
                                    expert_scores: Dict[str, np.ndarray],
                                    use_dynamic: Optional[bool] = None,
                                    variation_range: float = 0.2,
                                    steps: int = 10) -> Dict[str, Any]:
        """
        单因素敏感性分析，评估权重变化对风险评估的影响
        
        Args:
            factor_weights (Dict[str, float]): 因素权重
            expert_scores (Dict[str, np.ndarray]): 专家评分
            use_dynamic (Optional[bool]): 是否使用动态隶属度函数
            variation_range (float): 权重变化范围 (±)
            steps (int): 变化步数
            
        Returns:
            Dict[str, Any]: 敏感性分析结果
                {
                    "sensitivity_indices": {因素: 敏感性指标},
                    "variation_curves": {因素: {"variations": [变化率], "risk_indices": [风险指数]}},
                    "ranked_factors": [按敏感性排序的因素列表],
                    "baseline_result": 基准评价结果,
                    "critical_factors": [关键风险因素列表]
                }
        """
        # 计算基准评价结果
        baseline_result = self.evaluate(expert_scores, factor_weights, use_dynamic)
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
                modified_weights = {k: v/weight_sum for k, v in modified_weights.items()}
                
                # 使用修改后的权重重新计算风险评价
                result = self.evaluate(expert_scores, modified_weights, use_dynamic)
                risk_indices.append(result["risk_index"])
            
            # 计算敏感性指标 (风险指数变化率与权重变化率之比)
            # 使用线性回归斜率计算敏感性
            slope = np.polyfit(variations, risk_indices, 1)[0]
            sensitivity = slope * original_weight / baseline_risk_index
            
            # 存储结果
            sensitivity_indices[factor] = sensitivity
            variation_curves[factor] = {
                "variations": variations,
                "risk_indices": risk_indices
            }
        
        # 按敏感性排序因素
        ranked_factors = sorted(sensitivity_indices.items(), key=lambda x: abs(x[1]), reverse=True)
        ranked_factor_names = [item[0] for item in ranked_factors]
        
        # 识别关键风险因素 (敏感性指标大于平均值的因素)
        mean_sensitivity = np.mean([abs(s) for s in sensitivity_indices.values()])
        critical_factors = [factor for factor, sens in ranked_factors if abs(sens) > mean_sensitivity]
        
        # 整合结果
        result = {
            "sensitivity_indices": {f: sensitivity_indices[f] for f in ranked_factor_names},
            "variation_curves": variation_curves,
            "ranked_factors": ranked_factor_names,
            "baseline_result": baseline_result,
            "critical_factors": critical_factors,
            "mean_sensitivity": mean_sensitivity
        }
        
        # 缓存结果
        self.evaluation_cache["latest_sensitivity"] = result
        
        return result

    def _calculate_cross_sensitivity_matrix(self,
                                            factor1_values: np.ndarray,
                                            factor2_values: np.ndarray,
                                            baseline_risk: float,
                                            variation_steps: int) -> np.ndarray:
        """
        计算交叉敏感性矩阵，增强NaN处理能力

        Args:
            factor1_values (np.ndarray): 第一个因素的变化率数组
            factor2_values (np.ndarray): 第二个因素的变化率数组
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
                    # 修改当前权重配置
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

        Args:
            factor_weights (Dict[str, float]): 因素权重
            expert_scores (Dict[str, np.ndarray]): 专家评分
            factors (List[str]): 要分析的两个风险因素
            use_dynamic (Optional[bool]): 是否使用动态隶属度函数
            variation_range (float): 权重变化范围 (±)
            steps (int): 变化步数

        Returns:
            Dict[str, Any]: 交叉敏感性分析结果
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

        # 使用辅助方法生成敏感性矩阵
        risk_matrix = self._calculate_cross_sensitivity_matrix(
            [1 + v for v in variations],  # factor1的变化率
            [1 + v for v in variations],  # factor2的变化率
            baseline_risk_index,
            steps
        )

        # 整合结果
        result = {
            "risk_matrix": risk_matrix,
            "variations": variations,
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