# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from typing import List, Dict, Tuple, Optional, Callable, Any, Union
import logging
import os

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
    
    def generate_dynamic_membership_functions(self, expert_scores: np.ndarray) -> Dict[str, Callable]:
        """
        基于专家评分分布特征动态生成隶属度函数
        
        Args:
            expert_scores (np.ndarray): 专家评分数组(归一化后，0-1范围)
            
        Returns:
            Dict[str, Callable]: 动态生成的隶属度函数
        """
        # 计算评分的统计特性
        min_score = np.min(expert_scores)
        max_score = np.max(expert_scores)
        mean_score = np.mean(expert_scores)
        std_score = np.std(expert_scores)
        
        # 防止标准差过小导致的数值问题
        std_score = max(std_score, 0.05)
        
        # 计算分位数，用于更精确的分布刻画
        q1 = np.percentile(expert_scores, 25)  # 第一四分位数
        q2 = np.percentile(expert_scores, 50)  # 中位数
        q3 = np.percentile(expert_scores, 75)  # 第三四分位数
        
        # 根据评分分布特征构建梯形隶属度函数参数
        # 使用分位数和均值标准差结合的方式，使隶属度函数更好地适应数据分布
        dynamic_params = {
            "VL": (0, 0, min_score + 0.15*std_score, q1),
            "L": (min_score + 0.1*std_score, q1, q1 + 0.2*std_score, q2),
            "M": (q1, q2 - 0.15*std_score, q2 + 0.15*std_score, q3),
            "H": (q2, q3 - 0.2*std_score, q3, max_score - 0.1*std_score),
            "VH": (q3, max_score - 0.15*std_score, 1.0, 1.0)
        }
        
        # 参数合理性检查与调整
        for level, (a, b, c, d) in dynamic_params.items():
            # 确保参数在[0,1]范围内且满足单调递增
            a = max(0, min(a, 1))
            b = max(a, min(b, 1))
            c = max(b, min(c, 1))
            d = max(c, min(d, 1))
            dynamic_params[level] = (a, b, c, d)
            
            self.logger.debug(f"动态隶属度函数参数-{level}: a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f}")
        
        # 创建梯形隶属度函数
        return self._create_membership_functions(dynamic_params)
    
    def calculate_membership_degree(self, 
                                   expert_scores: np.ndarray, 
                                   use_dynamic: Optional[bool] = None) -> np.ndarray:
        """
        计算风险因素的隶属度向量
        
        Args:
            expert_scores (np.ndarray): 专家评分数组(归一化后，0-1范围)
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
            # 计算每个评分对当前风险等级的隶属度，然后取平均值
            membership_values = [mf(score) for score in scores]
            membership[i] = np.mean(membership_values)
        
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
                use_dynamic: Optional[bool] = None) -> Dict[str, Any]:
        """
        单层模糊综合评价
        
        Args:
            expert_scores (Dict[str, np.ndarray]): 专家评分 {因素: 评分数组}
            factor_weights (Dict[str, float]): 因素权重 {因素: 权重}
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
            membership = self.calculate_membership_degree(normalized_scores, use_dynamic)
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
    
    def cross_sensitivity_analysis(self,
                                  factor_weights: Dict[str, float],
                                  expert_scores: Dict[str, np.ndarray],
                                  factors: List[str],
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
                {
                    "risk_matrix": 风险指数矩阵,
                    "variations": 变化率列表,
                    "factors": 分析的风险因素,
                    "baseline_risk_index": 基准风险指数
                }
        """
        if len(factors) != 2:
            raise ValueError("交叉敏感性分析需要指定两个风险因素")
        
        # 确保因素存在于权重和评分中
        for factor in factors:
            if factor not in factor_weights or factor not in expert_scores:
                raise ValueError(f"风险因素 '{factor}' 在权重或评分中不存在")
        
        # 计算基准评价结果
        baseline_result = self.evaluate(expert_scores, factor_weights, use_dynamic)
        baseline_risk_index = baseline_result["risk_index"]
        
        # 初始化风险矩阵
        risk_matrix = np.zeros((steps + 1, steps + 1))
        
        # 计算权重变化步长和变化率列表
        step_size = 2 * variation_range / steps
        variations = [-variation_range + i * step_size for i in range(steps + 1)]
        
        # 提取两个因素
        factor1, factor2 = factors
        original_weight1 = factor_weights[factor1]
        original_weight2 = factor_weights[factor2]
        
        # 对两个因素同时进行变化分析
        for i, var1 in enumerate(variations):
            for j, var2 in enumerate(variations):
                # 创建修改后的权重字典
                modified_weights = copy.deepcopy(factor_weights)
                
                # 计算新权重 (确保不为负)
                new_weight1 = max(0.001, original_weight1 * (1 + var1))
                new_weight2 = max(0.001, original_weight2 * (1 + var2))
                
                modified_weights[factor1] = new_weight1
                modified_weights[factor2] = new_weight2
                
                # 归一化
                weight_sum = sum(modified_weights.values())
                modified_weights = {k: v/weight_sum for k, v in modified_weights.items()}
                
                # 使用修改后的权重重新计算风险评价
                result = self.evaluate(expert_scores, modified_weights, use_dynamic)
                risk_matrix[i, j] = result["risk_index"]
        
        # 整合结果
        result = {
            "risk_matrix": risk_matrix,
            "variations": variations,
            "factors": factors,
            "baseline_risk_index": baseline_risk_index
        }
        
        # 缓存结果
        self.evaluation_cache["latest_cross_sensitivity"] = result
        
        return result
    
    def visualize_membership_functions(self, 
                                      use_dynamic: bool = True, 
                                      scores: Optional[np.ndarray] = None,
                                      output_path: Optional[str] = None):
        """
        可视化隶属度函数
        
        Args:
            use_dynamic (bool): 是否使用动态隶属度函数
            scores (Optional[np.ndarray]): 用于生成动态隶属度函数的评分数据
            output_path (Optional[str]): 输出文件路径，如不提供则显示图形
        """
        # 确定使用哪种隶属度函数
        if use_dynamic:
            if scores is None:
                if self.last_used_membership_functions:
                    membership_functions = self.last_used_membership_functions
                    self.logger.info("使用最近一次的动态隶属度函数进行可视化")
                else:
                    self.logger.warning("未提供评分数据且无缓存的动态隶属度函数，将使用静态函数")
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
        
        plt.title("风险等级隶属度函数" + (" (动态)" if use_dynamic else " (静态)"))
        plt.xlabel("归一化评分")
        plt.ylabel("隶属度")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_sensitivity_analysis(self, 
                                      sensitivity_results: Optional[Dict[str, Any]] = None,
                                      top_n: int = 5,
                                      output_dir: Optional[str] = None):
        """
        可视化敏感性分析结果
        
        Args:
            sensitivity_results (Optional[Dict]): 敏感性分析结果，如不提供则使用最近缓存
            top_n (int): 显示前n个最敏感的因素
            output_dir (Optional[str]): 输出目录，如不提供则显示图形
        """
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
        
        # 1. 绘制敏感性指标条形图
        plt.figure(figsize=(12, 6))
        
        # 提取前top_n个因素数据
        top_factors = ranked_factors[:min(top_n, len(ranked_factors))]
        top_indices = [sensitivity_indices[f] for f in top_factors]
        
        # 为正负值设置不同颜色
        colors = ['#3498db' if v >= 0 else '#e74c3c' for v in top_indices]
        
        bars = plt.bar(top_factors, top_indices, color=colors)
        
        # 添加数值标签
        for bar, value in zip(bars, top_indices):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                    0.01 if height < 0 else height + 0.01,
                    f'{value:.4f}',
                    ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=9)
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axhline(y=sensitivity_results["mean_sensitivity"], color='green', 
                   linestyle='--', alpha=0.7, label=f'平均敏感性: {sensitivity_results["mean_sensitivity"]:.4f}')
        
        plt.title("风险因素敏感性分析")
        plt.xlabel("风险因素")
        plt.ylabel("敏感性指标")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "sensitivity_indices.png"), dpi=300)
            plt.close()
        else:
            plt.show()
        
        # 2. 绘制敏感性曲线图
        plt.figure(figsize=(12, 6))
        
        for factor in top_factors:
            curve = variation_curves[factor]
            plt.plot([v*100 for v in curve["variations"]], curve["risk_indices"], 
                    marker='o', linewidth=2, label=factor)
        
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.title("风险因素权重变化敏感性曲线")
        plt.xlabel("权重变化率 (%)")
        plt.ylabel("风险指数")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "sensitivity_curves.png"), dpi=300)
            plt.close()
        else:
            plt.show()
    
    def visualize_cross_sensitivity(self,
                                  cross_results: Optional[Dict[str, Any]] = None,
                                  output_dir: Optional[str] = None):
        """
        可视化交叉敏感性分析结果
        
        Args:
            cross_results (Optional[Dict]): 交叉敏感性分析结果，如不提供则使用最近缓存
            output_dir (Optional[str]): 输出目录，如不提供则显示图形
        """
        # 获取交叉敏感性分析结果
        if cross_results is None:
            if "latest_cross_sensitivity" in self.evaluation_cache:
                cross_results = self.evaluation_cache["latest_cross_sensitivity"]
            else:
                raise ValueError("未提供交叉敏感性分析结果且无缓存结果可用")
        
        # 提取数据
        risk_matrix = cross_results["risk_matrix"]
        variations = [f"{v*100:.0f}%" for v in cross_results["variations"]]
        factors = cross_results["factors"]
        baseline = cross_results["baseline_risk_index"]
        
        # 如果需要保存图片，确保目录存在
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 绘制热力图
        plt.figure(figsize=(10, 8))
        
        # 创建热力图
        sns.heatmap(risk_matrix, annot=True, fmt=".3f", cmap="viridis",
                   xticklabels=variations, yticklabels=variations)
        
        plt.title(f"交叉敏感性分析: {factors[0]} vs {factors[1]}")
        plt.xlabel(f"{factors[1]} 权重变化率")
        plt.ylabel(f"{factors[0]} 权重变化率")
        
        # 添加基准线
        mid_idx = len(variations) // 2
        plt.axvline(x=mid_idx, color='red', linestyle='--', alpha=0.7)
        plt.axhline(y=mid_idx, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "cross_sensitivity.png"), dpi=300)
            plt.close()
        else:
            plt.show()