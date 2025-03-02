# -*- coding: utf-8 -*-
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Tuple, Optional

import numpy as np

from visualizer import SensitivityVisualizer

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler()])


class FuzzyEvaluator:
    """模糊综合评价器，支持动态风险等级、动态隶属度函数和敏感性分析"""

    def __init__(self, risk_levels: List[str] = None):
        """初始化增强型模糊评价器"""
        self.risk_levels = risk_levels or ["VL", "L", "M", "H", "VH"]
        self.membership_functions = None

    def dynamic_membership_functions(self, expert_scores: np.ndarray) -> Dict[str, callable]:
        """
        基于专家评分分布特征动态构建隶属度函数

        Args:
            expert_scores (np.ndarray): 专家评分数组(归一化后，0-1范围)

        Returns:
            Dict[str, callable]: 动态调整后的隶属度函数字典
        """
        # 计算评分的统计特性
        min_score = np.min(expert_scores)
        max_score = np.max(expert_scores)
        mean_score = np.mean(expert_scores)
        std_score = np.std(expert_scores)

        # 基于数据分布特征定义梯形隶属度函数参数
        # 每个风险等级对应一个四元组(a,b,c,d)定义梯形函数
        dynamic_params = {
            "VL": (0, 0, min_score + 0.1 * std_score, mean_score - 0.8 * std_score),
            "L": (min_score + 0.1 * std_score, mean_score - 0.5 * std_score,
                  mean_score - 0.3 * std_score, mean_score),
            "M": (mean_score - 0.4 * std_score, mean_score - 0.1 * std_score,
                  mean_score + 0.1 * std_score, mean_score + 0.4 * std_score),
            "H": (mean_score, mean_score + 0.3 * std_score,
                  mean_score + 0.5 * std_score, max_score - 0.1 * std_score),
            "VH": (mean_score + 0.8 * std_score, max_score - 0.1 * std_score, 1.0, 1.0)
        }

        # 创建动态隶属度函数
        membership_functions = {}
        for level, params in dynamic_params.items():
            # 确保参数合理性(单调递增且在[0,1]范围内)
            a, b, c, d = params
            a = max(0, min(a, 1))
            b = max(a, min(b, 1))
            c = max(b, min(c, 1))
            d = max(c, min(d, 1))

            # 定义梯形隶属度函数
            def create_trap_mf(a, b, c, d):
                def trap_mf(x):
                    if x <= a or x >= d:
                        return 0.0
                    elif a < x <= b:
                        return (x - a) / (b - a + 1e-10)
                    elif b < x <= c:
                        return 1.0
                    else:  # c < x < d
                        return (d - x) / (d - c + 1e-10)

                return trap_mf

            membership_functions[level] = create_trap_mf(a, b, c, d)

        return membership_functions

    def calculate_membership_degree(self, expert_scores: np.ndarray, use_dynamic: bool = True) -> np.ndarray:
        """
        计算风险因素的隶属度向量

        Args:
            expert_scores (np.ndarray): 专家评分数组(归一化后，0-1范围)
            use_dynamic (bool): 是否使用动态隶属度函数

        Returns:
            np.ndarray: 隶属度向量
        """
        # 动态生成隶属度函数(每次评估都基于当前数据特征)
        if use_dynamic or self.membership_functions is None:
            self.membership_functions = self.dynamic_membership_functions(expert_scores)

        # 计算每个风险等级的隶属度
        membership = np.zeros(len(self.risk_levels))
        for i, level in enumerate(self.risk_levels):
            mf = self.membership_functions[level]
            membership[i] = np.mean([mf(score) for score in expert_scores])

        # 归一化处理
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

    def calculate_factor_risk_sensitivity_matrix(self, expert_scores_df, weight_vector: np.ndarray,
                                                 membership_matrix: np.ndarray) -> np.ndarray:
        """计算因素-风险等级敏感度矩阵"""
        num_factors = len(expert_scores_df.index)
        sensitivity_matrix = np.zeros((num_factors, len(self.risk_levels)))

        for i, factor_name in enumerate(expert_scores_df.index):
            perturbed_weight = weight_vector.copy()
            original_value = perturbed_weight[i]

            # 增加10%
            perturbed_weight[i] *= 1.1
            perturbed_weight = perturbed_weight / np.sum(perturbed_weight)
            high_result = self.fuzzy_multiply(perturbed_weight, membership_matrix)

            # 恢复并减少10%
            perturbed_weight[i] = original_value * 0.9
            perturbed_weight = perturbed_weight / np.sum(perturbed_weight)
            low_result = self.fuzzy_multiply(perturbed_weight, membership_matrix)

            # 计算敏感度
            baseline_result = self.fuzzy_multiply(weight_vector, membership_matrix)
            if original_value == 0:
                logging.warning(f"Skipping sensitivity calculation for factor '{factor_name}' due to zero weight.")
                sensitivity_matrix[i] = 0  # Avoid division by zero, set sensitivity to 0
            else:
                high_sensitivity = np.abs(high_result - baseline_result) / (0.1 * original_value)
                low_sensitivity = np.abs(low_result - baseline_result) / (0.1 * original_value)

                # 取平均敏感度
                sensitivity_matrix[i] = (high_sensitivity + low_sensitivity) / 2

        return sensitivity_matrix

    def generate_sensitivity_heatmap(self, sensitivity_matrix: np.ndarray, factor_names: List[str],
                                     output_path: str) -> None:
        """生成因素-风险等级敏感度热力图"""
        visualizer = SensitivityVisualizer()
        visualizer.plot_heatmap(
            sensitivity_matrix,
            row_labels=factor_names,
            col_labels=self.risk_levels,
            title="因素-风险等级敏感度热力图",
            output_path=output_path
        )

    def ensure_finite_data(data_dict, default_value=0.0):
        """
        确保数据字典中的所有数值均为有限值

        Args:
            data_dict: 数据字典
            default_value: 替换非有限值的默认值

        Returns:
            Dict: 处理后的数据字典
        """
        clean_dict = {}

        for key, value in data_dict.items():
            if isinstance(value, dict):
                clean_dict[key] = ensure_finite_data(value, default_value)
            elif isinstance(value, (list, tuple, np.ndarray)):
                if isinstance(value, np.ndarray):
                    # 处理NumPy数组
                    clean_array = value.copy()
                    mask = ~np.isfinite(clean_array)
                    if mask.any():
                        logging.warning(f"键 '{key}' 对应的数组包含 {mask.sum()} 个非有限值")
                        clean_array[mask] = default_value
                    clean_dict[key] = clean_array
                else:
                    # 处理列表或元组
                    clean_list = []
                    for item in value:
                        if np.isfinite(item) if isinstance(item, (int, float)) else True:
                            clean_list.append(item)
                        else:
                            logging.warning(f"键 '{key}' 对应的列表包含非有限值")
                            clean_list.append(default_value)
                    clean_dict[key] = type(value)(clean_list)  # 保持原始类型
            elif isinstance(value, (int, float)):
                if np.isfinite(value):
                    clean_dict[key] = value
                else:
                    logging.warning(f"键 '{key}' 对应的值 {value} 是非有限值，将被替换为 {default_value}")
                    clean_dict[key] = default_value
            else:
                clean_dict[key] = value

        return clean_dict

    def perform_fuzzy_evaluation_with_sensitivity(
            self, fuzzy_global_weights: Dict[str, float],
            fuzzy_excel_path: str,
            excel_handler,
            conduct_sensitivity: bool = False,
            sensitivity_config: Dict = None
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """执行模糊综合评价并进行敏感度分析

        Args:
            fuzzy_global_weights (Dict[str, float]): 全局权重字典
            fuzzy_excel_path (str): 模糊评价Excel文件路径
            excel_handler: Excel数据处理器实例
            conduct_sensitivity (bool): 是否进行敏感度分析
            sensitivity_config (Dict): 敏感度分析配置参数

        Returns:
            Tuple[Optional[np.ndarray], Optional[Dict]]:
                - 模糊评价结果向量
                - 敏感度分析结果（如果已进行）
        """
        try:
            # 执行标准模糊评价
            expert_scores_df, expert_weight = excel_handler.read_expert_scores(fuzzy_excel_path)
            if expert_weight is None:
                expert_weight = [1.0 / expert_scores_df.shape[1]] * expert_scores_df.shape[1]
            normalized_scores = expert_scores_df / 10.0
            # 获取专家数量和评价因素数量
            num_factors = len(expert_scores_df.index)
            num_experts = expert_scores_df.shape[1]
            # 初始化隶属度矩阵
            membership_matrix = np.zeros((num_factors, len(self.risk_levels)))
            # 对每个因素计算隶属度
            for i, factor in enumerate(expert_scores_df.index):
                factor_membership = np.zeros(len(self.risk_levels))
                # 分别计算每个专家对该因素的隶属度，然后按专家权重加权
                for j in range(num_experts):
                    # 获取单个专家对该因素的评分
                    expert_score = normalized_scores.loc[factor].iloc[j]
                    # 计算该专家评分的隶属度
                    expert_membership = self.calculate_membership_degree(np.array([expert_score]))
                    # 应用专家权重
                    factor_membership += expert_membership * expert_weight[j]
                # 归一化该因素的综合隶属度
                if np.sum(factor_membership) > 0:
                    factor_membership = factor_membership / np.sum(factor_membership)
                # 将该因素的隶属度向量存入隶属度矩阵
                membership_matrix[i] = factor_membership
            # 应用全局权重进行最终评价
            weight_vector = np.array([fuzzy_global_weights.get(factor, 0) for factor in expert_scores_df.index])
            fuzzy_result = self.fuzzy_multiply(weight_vector, membership_matrix)
            logging.info(f"模糊综合评价结论: {fuzzy_result}")

            # 执行敏感度分析（如果需要）
            sensitivity_results = None
            # 初始化可视化组件
            visualizer = SensitivityVisualizer()
            if conduct_sensitivity:
                config = sensitivity_config or {}
                perturbation_range = config.get('perturbation_range', 0.1)
                num_iterations = config.get('num_iterations', 100)
                thresholds = config.get('thresholds', {"VH": 0.7, "H": 0.6, "M": 0.5, "L": 0.4, "VL": 0.3})
                output_dir = config.get('output_dir', 'output/')
                # 权重敏感度分析
                weight_sensitivity = self.analyze_weight_sensitivity(weight_vector, membership_matrix,
                                                                     list(fuzzy_global_weights.keys()), num_iterations,
                                                                     perturbation_range)
                # 隶属度函数敏感度分析（随机选择一个因子）
                sample_idx = np.random.choice(normalized_scores.shape[0])
                sample_scores = normalized_scores.iloc[sample_idx].values
                membership_sensitivity = self.analyze_membership_sensitivity(sample_scores, num_iterations,
                                                                             perturbation_range)
                # 阈值影响分析
                threshold_impact = self.analyze_threshold_impact(weight_vector, membership_matrix, thresholds,
                                                                 num_iterations,
                                                                 perturbation_range)
                # 置信区间
                expert_scores_flat = normalized_scores.values.flatten()
                confidence_intervals = self.calculate_risk_confidence_interval(expert_scores_flat, weight_vector,
                                                                               membership_matrix)
                # 计算敏感度矩阵并生成热力图
                sensitivity_matrix = self.calculate_factor_risk_sensitivity_matrix(expert_scores_df, weight_vector,
                                                                                   membership_matrix)
                self.generate_sensitivity_heatmap(sensitivity_matrix, expert_scores_df.index.tolist(),
                                                  f"{output_dir}sensitivity_heatmap.png")

                sensitivity_results = {
                    'weight_sensitivity': weight_sensitivity,
                    'membership_sensitivity': membership_sensitivity,
                    'threshold_impact': threshold_impact,
                    'confidence_intervals': confidence_intervals,
                    'sensitivity_matrix': sensitivity_matrix
                }

            # 在处理敏感性分析数据前进行清洗
            sensitivity_results = ensure_finite_data(sensitivity_results)
            return fuzzy_result, sensitivity_results

        except Exception as e:
            logging.error(f"模糊综合评价出错: {str(e)}")
            return None, None

    def analyze_weight_sensitivity(self, weight_vector: np.ndarray, membership_matrix: np.ndarray,
                                   factor_names: List[str], num_iterations: int = 100,
                                   perturbation_range: float = 0.1) -> Dict[str, Dict[str, float]]:
        """分析全局权重敏感度"""
        baseline_result = self.fuzzy_multiply(weight_vector, membership_matrix)
        sensitivity_results = {name: {'direct_sensitivity': 0.0, 'relative_influence': 0.0, 'rank_stability': 0.0}
                               for name in factor_names}

        for idx, factor_name in enumerate(factor_names):
            if weight_vector[idx] <= 0:
                continue
            sensitivities, rank_changes = [], 0
            for _ in range(num_iterations):
                perturbed_weight = weight_vector.copy()
                perturbation = perturbed_weight[idx] * np.random.uniform(-perturbation_range, perturbation_range)
                perturbed_weight[idx] += perturbation
                perturbed_weight /= np.sum(perturbed_weight)
                perturbed_result = self.fuzzy_multiply(perturbed_weight, membership_matrix)
                result_change = np.abs(perturbed_result - baseline_result)
                if perturbation != 0:
                    sensitivity = np.mean(result_change) / (np.abs(perturbation) / weight_vector[idx])
                    sensitivities.append(sensitivity)
                if np.argmax(perturbed_result) != np.argmax(baseline_result):
                    rank_changes += 1
            if sensitivities:
                sensitivity_results[factor_name]['direct_sensitivity'] = np.mean(sensitivities)
                sensitivity_results[factor_name]['rank_stability'] = 1.0 - (rank_changes / num_iterations)

        total_sensitivity = sum(r['direct_sensitivity'] for r in sensitivity_results.values())
        if total_sensitivity > 0:
            for factor_name in sensitivity_results:
                sensitivity_results[factor_name]['relative_influence'] = sensitivity_results[factor_name][
                                                                             'direct_sensitivity'] / total_sensitivity
        return sensitivity_results

    def analyze_membership_sensitivity(self, expert_scores: np.ndarray, num_iterations: int = 100,
                                       perturbation_range: float = 0.1) -> Dict[str, Dict[str, float]]:
        """分析隶属度函数参数敏感度"""
        baseline_membership = self.calculate_membership_degree(expert_scores)
        sensitivity_results = {level: {'mean_variation': 0.0, 'std_deviation': 0.0, 'elasticity': 0.0}
                               for level in self.risk_levels}
        variations = []

        original_params = self.get_membership_parameters()
        for _ in range(num_iterations):
            perturbed_params = self._generate_perturbed_parameters(original_params, perturbation_range)
            self.set_membership_parameters(perturbed_params)
            perturbed_membership = self.calculate_membership_degree(expert_scores)
            self.set_membership_parameters(original_params)
            variations.append(np.abs(perturbed_membership - baseline_membership))

        variations_array = np.array(variations)
        for i, level in enumerate(self.risk_levels):
            sensitivity_results[level]['mean_variation'] = np.mean(variations_array[:, i])
            sensitivity_results[level]['std_deviation'] = np.std(variations_array[:, i])
            sensitivity_results[level]['elasticity'] = sensitivity_results[level]['mean_variation'] / perturbation_range
        return sensitivity_results

    def analyze_threshold_impact(self, weight_vector: np.ndarray, membership_matrix: np.ndarray,
                                 thresholds: Dict[str, float], num_iterations: int = 1000,
                                 perturbation_range: float = 0.1) -> Dict[str, float]:
        """分析阈值影响"""
        baseline_result = self.fuzzy_multiply(weight_vector, membership_matrix)
        baseline_category = self._determine_risk_category(baseline_result, thresholds)
        category_changes, risk_level_counts = 0, {level: 0 for level in self.risk_levels}

        for _ in range(num_iterations):
            perturbed_weight = self._generate_perturbed_weights(weight_vector, perturbation_range)
            perturbed_matrix = self._generate_perturbed_matrix(membership_matrix, perturbation_range)
            perturbed_result = self.fuzzy_multiply(perturbed_weight, perturbed_matrix)
            perturbed_category = self._determine_risk_category(perturbed_result, thresholds)
            if perturbed_category != baseline_category:
                category_changes += 1
            risk_level_counts[perturbed_category] += 1

        return {
            'category_change_probability': category_changes / num_iterations,
            'risk_distribution': {level: count / num_iterations for level, count in risk_level_counts.items()},
            'baseline_category': baseline_category
        }

    @staticmethod
    def simulation_worker(args):
        """Standalone simulation function for parallel processing"""
        idx, expert_scores, weight_vector, membership_matrix, evaluator = args
        # Generate perturbed samples
        perturbed_scores = evaluator._generate_perturbed_scores(expert_scores)
        perturbed_weights = evaluator._generate_perturbed_weights(weight_vector)

        # Calculate perturbed membership matrix
        perturbed_membership = np.zeros_like(membership_matrix)
        for j in range(membership_matrix.shape[0]):
            perturbed_membership[j] = evaluator.calculate_membership_degree(np.array([perturbed_scores[j]]))

        return evaluator.fuzzy_multiply(perturbed_weights, perturbed_membership)

    # Modified calculate_risk_confidence_interval method
    def calculate_risk_confidence_interval(self, expert_scores: np.ndarray, weight_vector: np.ndarray,
                                           membership_matrix: np.ndarray, confidence_level: float = 0.95,
                                           num_samples: int = 1000) -> Dict[str, Tuple[float, float]]:
        """计算风险评估置信区间"""
        # Create argument list for parallel tasks
        args = [(i, expert_scores, weight_vector, membership_matrix, self) for i in range(num_samples)]

        with ProcessPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(self.simulation_worker, args))

        results_array = np.array(results)
        alpha = (1 - confidence_level) / 2
        return {level: (np.percentile(results_array[:, i], 100 * alpha),
                        np.percentile(results_array[:, i], 100 * (1 - alpha)))
                for i, level in enumerate(self.risk_levels)}

    def _simulate_for_confidence_interval(self,
                                          idx,
                                          expert_scores,
                                          weight_vector,
                                          membership_matrix):
        """Simulation function for confidence interval calculation

        Args:
            idx: Index for random seed control (unused but required for mapping)
            expert_scores: Original expert scores
            weight_vector: Original weight vector
            membership_matrix: Original membership matrix

        Returns:
            np.ndarray: Simulated fuzzy evaluation result
        """
        # 生成扰动样本
        perturbed_scores = self._generate_perturbed_scores(expert_scores)
        perturbed_weights = self._generate_perturbed_weights(weight_vector)

        # 计算扰动后的隶属度矩阵
        perturbed_membership = np.zeros_like(membership_matrix)
        for j in range(membership_matrix.shape[0]):
            perturbed_membership[j] = self.calculate_membership_degree(np.array([perturbed_scores[j]]))

        return self.fuzzy_multiply(perturbed_weights, perturbed_membership)

    def _generate_perturbed_scores(self, scores: np.ndarray, perturbation_range: float = 0.05) -> np.ndarray:
        """生成扰动后的评分

        Args:
            scores (np.ndarray): 原始评分
            perturbation_range (float): 扰动范围

        Returns:
            np.ndarray: 扰动后的评分
        """
        perturbed = scores.copy()
        noise = np.random.uniform(-perturbation_range, perturbation_range, size=scores.shape)
        perturbed += noise
        # 确保评分在合理范围内
        return np.clip(perturbed, 0.0, 1.0)

    def _generate_perturbed_weights(self, weights: np.ndarray, perturbation_range: float = 0.05) -> np.ndarray:
        """生成扰动后的权重向量

        Args:
            weights (np.ndarray): 原始权重向量
            perturbation_range (float): 扰动范围

        Returns:
            np.ndarray: 扰动后的权重向量（已归一化）
        """
        perturbed = weights.copy()
        noise = np.random.uniform(-perturbation_range, perturbation_range, size=weights.shape)
        perturbed += noise * perturbed  # 扰动与权重成正比
        # 确保权重非负
        perturbed = np.maximum(0, perturbed)
        # 重新归一化
        return perturbed / np.sum(perturbed) if np.sum(perturbed) > 0 else perturbed

    def _generate_perturbed_matrix(self, matrix: np.ndarray, perturbation_range: float = 0.05) -> np.ndarray:
        perturbed = matrix.copy()
        for i in range(perturbed.shape[0]):
            for j in range(perturbed.shape[1]):
                if perturbed[i, j] > 0:
                    perturbed[i, j] *= (1 + np.random.uniform(-perturbation_range, perturbation_range))
            row_sum = np.sum(perturbed[i, :])
            if row_sum > 0:
                perturbed[i, :] /= row_sum
        return perturbed

    def _generate_perturbed_parameters(self,
                                       original_params: Dict[str, Tuple[float, float, float, float]],
                                       perturbation_range: float = 0.05) -> Dict[
        str, Tuple[float, float, float, float]]:
        """生成扰动后的隶属度函数参数

        Args:
            original_params (Dict[str, Tuple[float, float, float, float]]): 原始隶属度函数参数

        Returns:
            Dict[str, Tuple[float, float, float, float]]: 扰动后的隶属度函数参数
        """
        perturbed_params = {}

        for level, (a, b, c, d) in original_params.items():
            # 生成随机扰动因子
            rand_factor = 1.0 + np.random.uniform(-perturbation_range, perturbation_range)

            # 确保参数保持有效的梯形关系
            perturbed_a = a * rand_factor if a > 0 else a
            perturbed_b = max(perturbed_a, b * rand_factor)
            perturbed_c = max(perturbed_b, c * rand_factor)
            perturbed_d = max(perturbed_c, d * rand_factor if d < 1.0 else d)

            # 确保参数在[0,1]范围内
            perturbed_a = max(0, min(1, perturbed_a))
            perturbed_b = max(0, min(1, perturbed_b))
            perturbed_c = max(0, min(1, perturbed_c))
            perturbed_d = max(0, min(1, perturbed_d))

            perturbed_params[level] = (perturbed_a, perturbed_b, perturbed_c, perturbed_d)

        return perturbed_params

    def _determine_risk_category(self, result: np.ndarray, thresholds: Dict[str, float]) -> str:
        if not thresholds:
            return self.risk_levels[np.argmax(result)]
        for level, threshold in sorted(thresholds.items(), key=lambda x: x[1], reverse=True):
            if result[self.risk_levels.index(level)] >= threshold:
                return level
        return self.risk_levels[np.argmax(result)]

    def get_membership_parameters(self) -> Dict[str, Tuple[float, float, float, float]]:
        return self.membership_params.copy()

    def set_membership_parameters(self, params: Dict[str, Tuple[float, float, float, float]]) -> None:
        """设置隶属度函数参数并重新生成函数

        Args:
            params (Dict[str, Tuple[float, float, float, float]]): 新的隶属度函数参数
        """
        self._validate_membership_params(params)
        self.membership_params = params.copy()
