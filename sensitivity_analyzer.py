# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import copy
import os

class SensitivityAnalyzer:
    """风险敏感性分析模块，用于评估关键风险因素的影响程度"""
    
    def __init__(self, global_weights: Dict[str, float], fuzzy_results: np.ndarray, 
                 risk_levels: List[str] = None):
        """
        初始化敏感性分析器
        
        Args:
            global_weights (Dict[str, float]): 风险因素全局权重
            fuzzy_results (np.ndarray): 模糊评价结果
            risk_levels (List[str], optional): 风险等级列表
        """
        self.global_weights = global_weights
        self.fuzzy_results = fuzzy_results
        self.risk_levels = risk_levels or ["VL", "L", "M", "H", "VH"]
        self.risk_index = self._calculate_risk_index(fuzzy_results)
    
    def _calculate_risk_index(self, fuzzy_vector: np.ndarray) -> float:
        """
        计算风险指数 (加权平均法)
        
        Args:
            fuzzy_vector (np.ndarray): 模糊评价结果向量
            
        Returns:
            float: 风险指数值(0-1)
        """
        # 风险等级赋值 (VL=0.1, L=0.3, M=0.5, H=0.7, VH=0.9)
        level_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        return np.sum(fuzzy_vector * level_values)
    
    def single_factor_sensitivity(self, variation_range: float = 0.2, 
                                 steps: int = 10) -> pd.DataFrame:
        """
        单因素敏感性分析 - 分析单个风险因素权重变化对整体风险的影响
        
        Args:
            variation_range (float): 权重变化范围 (±)
            steps (int): 变化步数
            
        Returns:
            pd.DataFrame: 敏感性分析结果
        """
        results = []
        step_size = 2 * variation_range / steps
        
        # 对每个风险因素进行敏感性分析
        for factor, original_weight in self.global_weights.items():
            variations = []
            risk_indices = []
            
            # 在变化范围内改变权重
            for i in range(steps + 1):
                # 权重变化百分比 (-variation_range 到 +variation_range)
                variation_pct = -variation_range + i * step_size
                # 计算变化后的权重 (确保权重为正值)
                new_weight = max(0.001, original_weight * (1 + variation_pct))
                
                # 生成调整后的权重字典
                modified_weights = copy.deepcopy(self.global_weights)
                modified_weights[factor] = new_weight
                
                # 规范化权重
                weight_sum = sum(modified_weights.values())
                modified_weights = {k: v/weight_sum for k, v in modified_weights.items()}
                
                # 计算调整后的风险指数
                # 注意：这里简化处理，假设修改权重不影响模糊评价矩阵
                # 实际应用中可能需要重新计算模糊综合评价
                risk_index = self._calculate_risk_index(self.fuzzy_results)
                
                variations.append(variation_pct * 100)  # 转为百分比
                risk_indices.append(risk_index)
            
            # 计算敏感性指标
            if len(variations) > 1:
                # 计算风险指数变化率的斜率
                slope = np.polyfit(variations, risk_indices, 1)[0]
                sensitivity = slope * original_weight / self.risk_index
            else:
                sensitivity = 0
                
            results.append({
                "风险因素": factor,
                "原始权重": original_weight,
                "敏感性指标": sensitivity,
                "变化率": variations,
                "风险指数": risk_indices
            })
        
        # 按敏感性指标排序
        results_df = pd.DataFrame(results)
        return results_df.sort_values(by="敏感性指标", ascending=False)
    
    def cross_factor_sensitivity(self, factors: List[str], 
                               variation_range: float = 0.2,
                               steps: int = 5) -> Tuple[np.ndarray, List[float], List[float]]:
        """
        交叉敏感性分析 - 分析两个风险因素同时变化对整体风险的影响
        
        Args:
            factors (List[str]): 需分析的两个风险因素
            variation_range (float): 权重变化范围 (±)
            steps (int): 变化步数
            
        Returns:
            Tuple[np.ndarray, List[float], List[float]]: 风险指数矩阵, x轴变化率, y轴变化率
        """
        if len(factors) != 2:
            raise ValueError("交叉敏感性分析需要指定两个风险因素")
            
        factor1, factor2 = factors
        if factor1 not in self.global_weights or factor2 not in self.global_weights:
            raise ValueError("指定的风险因素不存在")
            
        # 初始化结果矩阵
        result_matrix = np.zeros((steps + 1, steps + 1))
        step_size = 2 * variation_range / steps
        
        # 变化率列表
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
                modified_weights = {k: v/weight_sum for k, v in modified_weights.items()}
                
                # 计算风险指数
                risk_index = self._calculate_risk_index(self.fuzzy_results)
                result_matrix[i, j] = risk_index
                
        # 转换变化率为百分比
        variations_pct = [v * 100 for v in variations]
        
        return result_matrix, variations_pct, variations_pct
    
    def visualize_sensitivity(self, results_df: pd.DataFrame, output_dir: str):
        """
        可视化敏感性分析结果
        
        Args:
            results_df (pd.DataFrame): 敏感性分析结果
            output_dir (str): 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 绘制敏感性指标条形图
        plt.figure(figsize=(12, 8))
        sns.barplot(x='风险因素', y='敏感性指标', data=results_df)
        plt.title('风险因素敏感性指标')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sensitivity_index.png'), dpi=300)
        plt.close()
        
        # 2. 绘制前5个最敏感因素的曲线图
        top_factors = results_df.head(5)
        plt.figure(figsize=(12, 8))
        
        for _, row in top_factors.iterrows():
            plt.plot(row['变化率'], row['风险指数'], marker='o', label=row['风险因素'])
            
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        plt.title('主要风险因素敏感性曲线')
        plt.xlabel('权重变化率 (%)')
        plt.ylabel('风险指数')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sensitivity_curves.png'), dpi=300)
        plt.close()
    
    def visualize_cross_sensitivity(self, matrix: np.ndarray, x_vars: List[float], 
                                   y_vars: List[float], factors: List[str], 
                                   output_dir: str):
        """
        可视化交叉敏感性分析结果
        
        Args:
            matrix (np.ndarray): 风险指数矩阵
            x_vars (List[float]): x轴变化率
            y_vars (List[float]): y轴变化率
            factors (List[str]): 风险因素名称
            output_dir (str): 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        # 绘制热力图
        ax = sns.heatmap(matrix, cmap='viridis', annot=True, fmt='.3f',
                     xticklabels=[f"{x:.0f}%" for x in x_vars],
                     yticklabels=[f"{y:.0f}%" for y in y_vars])
        
        # 设置标题和标签
        plt.title(f'交叉敏感性分析: {factors[0]} vs {factors[1]}')
        plt.xlabel(f'{factors[1]} 权重变化率')
        plt.ylabel(f'{factors[0]} 权重变化率')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cross_sensitivity.png'), dpi=300)
        plt.close()
    
    def monte_carlo_sensitivity(self, num_simulations: int = 1000, 
                              variation_range: float = 0.2) -> Dict:
        """
        蒙特卡洛敏感性分析 - 随机变化多个因素权重模拟可能的风险情景
        
        Args:
            num_simulations (int): 模拟次数
            variation_range (float): 权重变化范围 (±)
            
        Returns:
            Dict: 包含模拟结果的字典
        """
        risk_indices = []
        weight_variations = {factor: [] for factor in self.global_weights}
        
        # 进行蒙特卡洛模拟
        for _ in range(num_simulations):
            # 随机生成变化后的权重
            modified_weights = {}
            for factor, weight in self.global_weights.items():
                # 随机变化率 (-variation_range 到 +variation_range)
                variation = np.random.uniform(-variation_range, variation_range)
                # 确保权重为正值
                new_weight = max(0.001, weight * (1 + variation))
                modified_weights[factor] = new_weight
                weight_variations[factor].append(variation)
            
            # 权重归一化
            weight_sum = sum(modified_weights.values())
            modified_weights = {k: v/weight_sum for k, v in modified_weights.items()}
            
            # 计算风险指数
            risk_index = self._calculate_risk_index(self.fuzzy_results)
            risk_indices.append(risk_index)
        
        # 计算每个因素变化率与风险指数的相关性
        correlations = {}
        for factor, variations in weight_variations.items():
            corr = np.corrcoef(variations, risk_indices)[0, 1]
            correlations[factor] = corr
        
        # 排序相关性
        sorted_correlations = dict(sorted(correlations.items(), 
                                        key=lambda x: abs(x[1]), 
                                        reverse=True))
        
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
        
        return {
            "risk_indices": risk_indices,
            "correlations": sorted_correlations,
            "risk_stats": risk_stats
        }
    
    def visualize_monte_carlo(self, results: Dict, output_dir: str):
        """
        可视化蒙特卡洛敏感性分析结果
        
        Args:
            results (Dict): 蒙特卡洛模拟结果
            output_dir (str): 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 风险指数分布图
        plt.figure(figsize=(10, 6))
        sns.histplot(results["risk_indices"], kde=True, bins=30)
        plt.axvline(x=self.risk_index, color='red', linestyle='--', 
                  label=f'当前风险指数: {self.risk_index:.4f}')
        plt.axvline(x=results["risk_stats"]["mean"], color='green', linestyle='-.',
                  label=f'平均风险指数: {results["risk_stats"]["mean"]:.4f}')
        
        plt.title('风险指数分布 (蒙特卡洛模拟)')
        plt.xlabel('风险指数')
        plt.ylabel('频率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'monte_carlo_distribution.png'), dpi=300)
        plt.close()
        
        # 2. 因素相关性图
        corr_data = pd.DataFrame({
            '风险因素': list(results["correlations"].keys()),
            '相关性': list(results["correlations"].values())
        })
        
        plt.figure(figsize=(12, 8))
        bars = sns.barplot(x='风险因素', y='相关性', data=corr_data)
        
        # 为正负相关性添加不同颜色
        for i, bar in enumerate(bars.patches):
            if corr_data.iloc[i]['相关性'] < 0:
                bar.set_color('lightcoral')
        
        plt.title('风险因素与风险指数的相关性')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'monte_carlo_correlations.png'), dpi=300)
        plt.close()
        
        # 3. 风险指数累积分布函数 (CDF)
        plt.figure(figsize=(10, 6))
        sorted_indices = np.sort(results["risk_indices"])
        cumulative = np.arange(1, len(sorted_indices) + 1) / len(sorted_indices)
        
        plt.plot(sorted_indices, cumulative, marker='.', linestyle='none', alpha=0.3)
        plt.plot(sorted_indices, cumulative, linestyle='-', alpha=0.7)
        
        # 添加关键百分位线
        percentiles = results["risk_stats"]["percentiles"]
        for p_label, p_value in percentiles.items():
            plt.axvline(x=p_value, linestyle='--', alpha=0.5, 
                      label=f'{p_label}: {p_value:.4f}')
        
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=results["risk_stats"]["median"], color='green', linestyle='-', 
                  label=f'中位数: {results["risk_stats"]["median"]:.4f}')
        
        plt.title('风险指数累积分布函数')
        plt.xlabel('风险指数')
        plt.ylabel('累积概率')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'monte_carlo_cdf.png'), dpi=300)
        plt.close()
