# risk_sensitivity_analyzer.py
import copy
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RiskSensitivityAnalyzer:
    """风险敏感性分析器，用于评估风险因素的敏感度和影响"""

    def __init__(self, global_weights: Dict[str, float], fuzzy_result: np.ndarray,
                 risk_levels: List[str] = None):
        """
        初始化风险敏感性分析器
        
        参数:
            global_weights (Dict[str, float]): 风险因素的全局权重字典
            fuzzy_result (np.ndarray): 模糊评价结果向量
            risk_levels (List[str], optional): 风险等级列表
        """
        self.global_weights = global_weights
        self.fuzzy_result = fuzzy_result
        self.risk_levels = risk_levels or ["VL", "L", "M", "H", "VH"]

        # 初始化风险指数
        self.risk_index = self._calculate_risk_index(fuzzy_result)

        # 记录敏感性分析结果
        self.sensitivity_results = {}

        # 配置日志
        self.logger = logging.getLogger(__name__)

    def _calculate_risk_index(self, fuzzy_vector: np.ndarray) -> float:
        """
        计算风险指数(加权平均法)
        
        参数:
            fuzzy_vector (np.ndarray): 模糊评价结果向量
            
        返回:
            float: 风险指数值(0-1)
        """
        # 风险等级对应的数值评分
        # VL=0.1, L=0.3, M=0.5, H=0.7, VH=0.9
        level_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # 计算加权平均
        risk_index = np.sum(fuzzy_vector * level_values)

        return risk_index

    def single_factor_sensitivity(self,
                                  variation_range: float = 0.2,
                                  steps: int = 10,
                                  membership_matrix: Optional[np.ndarray] = None,
                                  factor_evaluator=None) -> Dict[str, Any]:
        """
        单因素敏感性分析，评估单个风险因素权重变化对整体风险的影响
        
        参数:
            variation_range (float): 权重变化范围 (±)
            steps (int): 变化步数
            membership_matrix (Optional[np.ndarray]): 隶属度矩阵
            factor_evaluator: 模糊评价器引用，用于重新计算结果
            
        返回:
            Dict[str, Any]: 敏感性分析结果
        """
        results = {}
        factor_sensitivities = {}
        variation_curves = {}
        step_size = 2 * variation_range / steps

        # 对每个风险因素进行敏感性分析
        for factor, original_weight in self.global_weights.items():
            variations = []
            risk_indices = []

            # 在变化范围内改变权重
            for i in range(steps + 1):
                # 权重变化百分比(-variation_range 到 +variation_range)
                variation_pct = -variation_range + i * step_size

                # 计算变化后的权重(确保权重为正值)
                new_weight = max(0.001, original_weight * (1 + variation_pct))

                # 生成调整后的权重字典
                modified_weights = copy.deepcopy(self.global_weights)
                modified_weights[factor] = new_weight

                # 规范化权重
                weight_sum = sum(modified_weights.values())
                if weight_sum > 0:
                    modified_weights = {k: v / weight_sum for k, v in modified_weights.items()}

                # 计算调整后的风险指数
                if membership_matrix is not None and factor_evaluator is not None:
                    # 使用模糊评价器重新计算
                    weight_vector = np.array([modified_weights.get(f, 0) for f in membership_matrix.index])
                    new_fuzzy_result = factor_evaluator.fuzzy_multiply(weight_vector, membership_matrix.values)
                    risk_index = self._calculate_risk_index(new_fuzzy_result)
                else:
                    # 简化处理，假设修改权重不影响模糊评价矩阵
                    risk_index = self._calculate_risk_index(self.fuzzy_result)

                variations.append(variation_pct)
                risk_indices.append(risk_index)

            # 计算敏感性指标
            if len(variations) > 1:
                # 使用线性回归计算斜率作为敏感性指标
                slope, _ = np.polyfit(variations, risk_indices, 1)

                # 标准化敏感性指标
                if original_weight > 0 and self.risk_index > 0:
                    sensitivity = slope * original_weight / self.risk_index
                else:
                    sensitivity = 0
            else:
                sensitivity = 0

            # 存储敏感性结果
            factor_sensitivities[factor] = sensitivity
            variation_curves[factor] = {
                "variations": variations,
                "risk_indices": risk_indices
            }

        # 对敏感性指标进行排序
        ranked_factors = sorted(
            factor_sensitivities.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # 从排序结果中提取因素名称
        ranked_factor_names = [item[0] for item in ranked_factors]

        # 计算平均敏感性
        mean_sensitivity = np.mean([abs(v) for v in factor_sensitivities.values()])

        # 识别关键风险因素(敏感性大于平均值的因素)
        critical_factors = [
            factor for factor, sens in ranked_factors
            if abs(sens) > mean_sensitivity
        ]

        # 构建结果字典
        results = {
            "sensitivity_indices": factor_sensitivities,
            "variation_curves": variation_curves,
            "ranked_factors": ranked_factor_names,
            "critical_factors": critical_factors,
            "mean_sensitivity": mean_sensitivity
        }

        # 记录结果
        self.sensitivity_results['single_factor'] = results

        return results

    def cross_factor_sensitivity(self,
                                 factors: List[str],
                                 variation_range: float = 0.2,
                                 steps: int = 5,
                                 membership_matrix: Optional[np.ndarray] = None,
                                 factor_evaluator=None) -> Dict[str, Any]:
        """
        交叉敏感性分析，评估两个风险因素同时变化对整体风险的影响
        
        参数:
            factors (List[str]): 需要分析的风险因素(两个)
            variation_range (float): 权重变化范围(±)
            steps (int): 变化步数
            membership_matrix (Optional[np.ndarray]): 隶属度矩阵
            factor_evaluator: 模糊评价器引用
            
        返回:
            Dict[str, Any]: 交叉敏感性分析结果
        """
        if len(factors) != 2:
            raise ValueError("交叉敏感性分析需要指定两个风险因素")

        factor1, factor2 = factors
        if factor1 not in self.global_weights or factor2 not in self.global_weights:
            raise ValueError(f"指定的风险因素不存在: {factor1}, {factor2}")

        # 初始化结果矩阵
        risk_matrix = np.zeros((steps + 1, steps + 1))

        # 计算步长和变化率列表
        step_size = 2 * variation_range / steps
        variations = [-variation_range + i * step_size for i in range(steps + 1)]

        # 对两个因素同时进行敏感性分析
        for i, var1 in enumerate(variations):
            for j, var2 in enumerate(variations):
                # 修改权重
                modified_weights = copy.deepcopy(self.global_weights)

                # 计算新权重
                weight1 = max(0.001, self.global_weights[factor1] * (1 + var1))
                weight2 = max(0.001, self.global_weights[factor2] * (1 + var2))

                modified_weights[factor1] = weight1
                modified_weights[factor2] = weight2

                # 权重归一化
                weight_sum = sum(modified_weights.values())
                if weight_sum > 0:
                    modified_weights = {k: v / weight_sum for k, v in modified_weights.items()}

                # 计算风险指数
                if membership_matrix is not None and factor_evaluator is not None:
                    # 使用模糊评价器重新计算
                    weight_vector = np.array([modified_weights.get(f, 0) for f in membership_matrix.index])
                    new_fuzzy_result = factor_evaluator.fuzzy_multiply(weight_vector, membership_matrix.values)
                    risk_index = self._calculate_risk_index(new_fuzzy_result)
                else:
                    # 简化处理
                    risk_index = self._calculate_risk_index(self.fuzzy_result)

                risk_matrix[i, j] = risk_index

        # 构建结果字典
        results = {
            "risk_matrix": risk_matrix,
            "variations": variations,
            "factors": factors,
            "baseline_risk_index": self.risk_index
        }

        # 记录结果
        self.sensitivity_results['cross_factor'] = results

        return results

    def monte_carlo_sensitivity(self,
                                num_simulations: int = 1000,
                                variation_range: float = 0.2,
                                membership_matrix: Optional[np.ndarray] = None,
                                factor_evaluator=None) -> Dict[str, Any]:
        """
        蒙特卡洛敏感性分析，通过随机变化多个因素权重模拟风险情景
        
        参数:
            num_simulations (int): 模拟次数
            variation_range (float): 权重变化范围(±)
            membership_matrix (Optional[np.ndarray]): 隶属度矩阵
            factor_evaluator: 模糊评价器引用
            
        返回:
            Dict[str, Any]: 蒙特卡洛敏感性分析结果
        """
        # 初始化结果容器
        risk_indices = []
        weight_variations = {factor: [] for factor in self.global_weights}
        simulation_weights = []

        # 使用ProcessPoolExecutor进行并行计算
        if membership_matrix is not None and factor_evaluator is not None:
            with ProcessPoolExecutor(max_workers=4) as executor:
                # 创建模拟参数
                params = []
                for i in range(num_simulations):
                    # 随机生成变化后的权重
                    modified_weights = {}
                    variations = {}

                    for factor, weight in self.global_weights.items():
                        # 随机变化率(-variation_range 到 +variation_range)
                        variation = np.random.uniform(-variation_range, variation_range)
                        variations[factor] = variation

                        # 确保权重为正值
                        new_weight = max(0.001, weight * (1 + variation))
                        modified_weights[factor] = new_weight

                    # 权重归一化
                    weight_sum = sum(modified_weights.values())
                    if weight_sum > 0:
                        modified_weights = {k: v / weight_sum for k, v in modified_weights.items()}

                    params.append((modified_weights, variations, membership_matrix, factor_evaluator))

                # 并行执行模拟
                results = list(executor.map(self._monte_carlo_worker, params))

                # 处理结果
                for result in results:
                    risk_index, variations, modified_weights = result
                    risk_indices.append(risk_index)
                    simulation_weights.append(modified_weights)

                    # 记录每个因素的变化率
                    for factor, variation in variations.items():
                        weight_variations[factor].append(variation)
        else:
            # 如果没有提供隶属度矩阵或评价器，使用串行计算
            for i in range(num_simulations):
                # 随机生成变化后的权重
                modified_weights = {}

                for factor, weight in self.global_weights.items():
                    # 随机变化率(-variation_range 到 +variation_range)
                    variation = np.random.uniform(-variation_range, variation_range)
                    weight_variations[factor].append(variation)

                    # 确保权重为正值
                    new_weight = max(0.001, weight * (1 + variation))
                    modified_weights[factor] = new_weight

                # 权重归一化
                weight_sum = sum(modified_weights.values())
                if weight_sum > 0:
                    modified_weights = {k: v / weight_sum for k, v in modified_weights.items()}

                simulation_weights.append(modified_weights)

                # 简化处理，使用原始模糊评价结果
                risk_index = self._calculate_risk_index(self.fuzzy_result)
                risk_indices.append(risk_index)

        # 计算每个因素变化率与风险指数的相关性
        correlations = {}
        for factor, variations in weight_variations.items():
            corr = np.corrcoef(variations, risk_indices)[0, 1]
            correlations[factor] = corr

        # 对相关性进行排序
        sorted_correlations = dict(sorted(
            correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))

        # 计算风险指数统计信息
        risk_stats = {
            "mean": np.mean(risk_indices),
            "median": np.median(risk_indices),
            "std": np.std(risk_indices),
            "min": np.min(risk_indices),
            "max": np.max(risk_indices),
            "percentiles": {
                "5%": np.percentile(risk_indices, 5),
                "25%": np.percentile(risk_indices, 25),
                "75%": np.percentile(risk_indices, 75),
                "95%": np.percentile(risk_indices, 95)
            }
        }

        # 构建返回结果
        results = {
            "risk_indices": risk_indices,
            "correlations": sorted_correlations,
            "risk_stats": risk_stats,
            "weight_variations": weight_variations,
            "simulation_weights": simulation_weights
        }

        # 记录结果
        self.sensitivity_results['monte_carlo'] = results

        return results

    def _monte_carlo_worker(self, params):
        """
        蒙特卡洛模拟工作函数，用于并行计算
        
        参数:
            params: 包含模拟参数的元组
            
        返回:
            Tuple: (风险指数, 变化率字典, 修改后的权重字典)
        """
        modified_weights, variations, membership_matrix, factor_evaluator = params

        # 计算风险指数
        weight_vector = np.array([modified_weights.get(f, 0) for f in membership_matrix.index])
        new_fuzzy_result = factor_evaluator.fuzzy_multiply(weight_vector, membership_matrix.values)
        risk_index = self._calculate_risk_index(new_fuzzy_result)

        return risk_index, variations, modified_weights

    def calculate_threshold_impact(self,
                                   num_simulations: int = 1000,
                                   thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        """
        计算风险阈值影响，评估风险等级分类的稳定性
        
        参数:
            num_simulations (int): 模拟次数
            thresholds (Dict[str, float]): 风险等级阈值，格式:{等级: 阈值}
            
        返回:
            Dict[str, Any]: 阈值影响分析结果
        """
        # 默认阈值
        if thresholds is None:
            thresholds = {
                "VL": 0.3,
                "L": 0.4,
                "M": 0.5,
                "H": 0.6,
                "VH": 0.7
            }

        # 确定基准风险等级
        baseline_category = self._determine_risk_category(self.fuzzy_result, thresholds)

        # 如果没有蒙特卡洛模拟结果，返回简化结果
        if 'monte_carlo' not in self.sensitivity_results:
            return {
                "baseline_category": baseline_category,
                "category_change_probability": 0,
                "risk_distribution": {level: 0 for level in self.risk_levels}
            }

        # 获取蒙特卡洛模拟结果
        mc_results = self.sensitivity_results['monte_carlo']
        risk_indices = mc_results["risk_indices"]

        # 初始化计数器
        category_changes = 0
        risk_level_counts = {level: 0 for level in self.risk_levels}

        # 分析每个模拟结果
        for risk_index in risk_indices:
            # 将风险指数转换为风险等级
            category = self._determine_category_from_index(risk_index, thresholds)

            # 统计风险等级变化
            if category != baseline_category:
                category_changes += 1

            # 统计各风险等级出现次数
            risk_level_counts[category] += 1

        # 计算风险等级分布概率
        risk_distribution = {
            level: count / num_simulations
            for level, count in risk_level_counts.items()
        }

        # 构建结果字典
        results = {
            "baseline_category": baseline_category,
            "category_change_probability": category_changes / num_simulations,
            "risk_distribution": risk_distribution
        }

        return results

    def _determine_risk_category(self, fuzzy_result: np.ndarray,
                                 thresholds: Dict[str, float]) -> str:
        """
        根据模糊评价结果和阈值确定风险等级
        
        参数:
            fuzzy_result (np.ndarray): 模糊评价结果向量
            thresholds (Dict[str, float]): 风险等级阈值
            
        返回:
            str: 风险等级
        """
        # 如果没有提供阈值，使用最大隶属度法
        if not thresholds:
            return self.risk_levels[np.argmax(fuzzy_result)]

        # 使用阈值方法
        for level, threshold in sorted(thresholds.items(), key=lambda x: x[1], reverse=True):
            if fuzzy_result[self.risk_levels.index(level)] >= threshold:
                return level

        # 如果没有满足阈值的等级，使用最大隶属度法
        return self.risk_levels[np.argmax(fuzzy_result)]

    def _determine_category_from_index(self, risk_index: float,
                                       thresholds: Dict[str, float]) -> str:
        """
        根据风险指数确定风险等级
        
        参数:
            risk_index (float): 风险指数
            thresholds (Dict[str, float]): 风险等级阈值
            
        返回:
            str: 风险等级
        """
        # 简化的风险等级判定
        if risk_index < 0.3:
            return "VL"
        elif risk_index < 0.5:
            return "L"
        elif risk_index < 0.7:
            return "M"
        elif risk_index < 0.9:
            return "H"
        else:
            return "VH"

    def calculate_confidence_intervals(self,
                                       confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """
        计算模糊评价结果的置信区间
        
        参数:
            confidence_level (float): 置信水平，默认0.95
            
        返回:
            Dict[str, Tuple[float, float]]: 各风险等级的置信区间
        """
        # 如果没有蒙特卡洛模拟结果，返回空结果
        if 'monte_carlo' not in self.sensitivity_results:
            return {}

        # 获取蒙特卡洛模拟结果
        mc_results = self.sensitivity_results['monte_carlo']
        risk_indices = mc_results["risk_indices"]

        # 计算置信区间
        alpha = (1 - confidence_level) / 2
        lower_bound = np.percentile(risk_indices, 100 * alpha)
        upper_bound = np.percentile(risk_indices, 100 * (1 - alpha))

        # 返回结果
        return {
            "risk_index": (lower_bound, upper_bound)
        }
