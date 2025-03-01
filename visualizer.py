# visualizer.py
import platform
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch


class Visualizer:
    """结果可视化模块"""
    @staticmethod
    def get_chinese_font():
        """获取适合当前系统的中文字体"""
        system = platform.system()
        if system == 'Windows':
            return FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")
        elif system == 'Darwin':  # macOS
            return FontProperties(fname="/System/Library/Fonts/Hiragino Sans GB.ttc")
        else:  # Linux and others
            # 使用matplotlib内置的中文字体
            return FontProperties(family='sans-serif')

    @staticmethod
    def configure_chinese_fonts():
        """配置matplotlib支持中文显示"""
        system = platform.system()
        if system == 'Windows':
            plt.rcParams['font.sans-serif'] = ['SimHei']
        elif system == 'Darwin':  # macOS
            plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']
        else:  # Linux and others
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    @staticmethod
    def plot_weights_pie(weights_dict: Dict[str, float], title: str, output_path: str):
        """绘制权重分布的饼状图"""
        Visualizer.configure_chinese_fonts()
        plt.figure(figsize=(10, 8))  # Set figure size
        font_properties = Visualizer.get_chinese_font()
        labels = list(weights_dict.keys())
        weights = list(weights_dict.values())
        plt.pie(
            weights,
            labels=labels,
            autopct="%.1f%%",
            textprops={"fontproperties": font_properties, "fontsize": 12}
        )
        plt.title(title, fontproperties=font_properties, fontsize=14)
        plt.savefig(output_path, dpi=300)  # Save at high resolution
        plt.close()

    @staticmethod
    def plot_fuzzy_result(fuzzy_result: np.ndarray, output_path: str):
        """绘制模糊隶属度图"""
        Visualizer.configure_chinese_fonts()
        plt.figure(figsize=(10, 6))  # Increase figure size
        font_properties = Visualizer.get_chinese_font()
        plt.bar(["VL", "L", "M", "H", "VH"], fuzzy_result)
        plt.title("模糊综合评价结果", fontproperties=font_properties, fontsize=14)
        plt.xlabel("风险等级", fontproperties=font_properties, fontsize=14)
        plt.ylabel("隶属度", fontproperties=font_properties, fontsize=14)
        plt.savefig(output_path, dpi=300)  # Set DPI for higher resolution
        plt.close()


class SensitivityVisualizer:
    """敏感度分析可视化组件"""

    def __init__(self, risk_levels: List[str] = None, color_map: Dict[str, str] = None):
        """初始化敏感度可视化组件

        Args:
            risk_levels (List[str], optional): 风险等级列表
            color_map (Dict[str, str], optional): 风险等级颜色映射
        """
        self.risk_levels = risk_levels or ["VL", "L", "M", "H", "VH"]
        # 默认颜色映射: VL->绿色, L->浅绿, M->黄色, H->橙色, VH->红色
        self.color_map = color_map or {
            "VL": "#1a9850", "L": "#91cf60", "M": "#ffffbf",
            "H": "#fc8d59", "VH": "#d73027"
        }
        # 设置默认字体, 可视化样式
        self.font_properties = Visualizer.get_chinese_font()
        plt.style.use('seaborn-v0_8-whitegrid')
        self.default_figsize = (10, 6)
        # 配置中文字体
        Visualizer.configure_chinese_fonts()

    def plot_tornado_diagram(self,
                             sensitivity_data: Dict[str, Dict[str, float]],
                             metric: str = 'relative_influence',
                             title: str = "Factor Sensitivity Analysis",
                             output_path: Optional[str] = None) -> Figure:
        """绘制龙卷风图展示因素敏感度

        Args:
            sensitivity_data (Dict[str, Dict[str, float]]): 敏感度分析结果
            metric (str): 要展示的指标，如 'relative_influence', 'direct_sensitivity'
            title (str): 图表标题
            output_path (str, optional): 输出文档路径

        Returns:
            Figure: 生成的图表对象
        """
        # 提取数据
        factors = []
        values = []

        for factor, metrics in sensitivity_data.items():
            if metric in metrics:
                factors.append(factor)
                values.append(metrics[metric])

        # 数据排序
        sorted_indices = np.argsort(values)
        factors = [factors[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]

        # 创建图表
        fig, ax = plt.subplots(figsize=self.default_figsize)

        # 绘制水平条形图
        bars = ax.barh(factors, values, height=0.6)

        # 为条形图着色（基于值的大小）
        colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(values)))
        for i, bar in enumerate(bars):
            bar.set_color(colors[i])

        # 添加数值标签
        for i, v in enumerate(values):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center')

        # 设置标题和标签
        ax.set_title(title, fontproperties=self.font_properties, fontsize=14, pad=20)
        ax.set_xlabel(f'Sensitivity ({metric})', fontproperties=self.font_properties, fontsize=12)
        ax.set_ylabel('Factors', fontproperties=self.font_properties, fontsize=12)

        # 调整布局
        plt.tight_layout()

        # 保存图表
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        return fig

    def plot_heatmap(self,
                     matrix_data: np.ndarray,
                     row_labels: List[str],
                     col_labels: List[str],
                     title: str = "Sensitivity Heatmap",
                     output_path: Optional[str] = None,
                     annotate: bool = True) -> Figure:
        """绘制热力图展示敏感度分布

        Args:
            matrix_data (np.ndarray): 矩阵数据
            row_labels (List[str]): 行标签
            col_labels (List[str]): 列标签
            title (str): 图表标题
            output_path (str, optional): 输出文档路径
            annotate (bool): 是否在单元格中显示数值

        Returns:
            Figure: 生成的图表对象
        """
        # 创建图表
        fig, ax = plt.subplots(figsize=self.default_figsize)

        # 绘制热力图
        sns.heatmap(
            matrix_data,
            annot=annotate,
            fmt='.3f' if annotate else '',
            cmap='YlOrRd',
            linewidths=.5,
            ax=ax,
            xticklabels=col_labels,
            yticklabels=row_labels,
            cbar_kws={'label': 'Sensitivity Value'}
        )

        # 设置标题
        ax.set_title(title, fontproperties=self.font_properties, fontsize=14, pad=20)

        # 调整布局
        plt.tight_layout()

        # 保存图表
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        return fig

    def plot_membership_sensitivity(self,
                                   sensitivity_results: Dict[str, Dict[str, float]],
                                   metrics: List[str] = None,
                                   title: str = "Membership Function Sensitivity Analysis",
                                   output_path: Optional[str] = None) -> Figure:
        """绘制隶属度函数敏感度分析图

        Args:
            sensitivity_results (Dict[str, Dict[str, float]]): 敏感度分析结果
            metrics (List[str], optional): 要展示的指标列表
            title (str): 图表标题
            output_path (str, optional): 输出文档路径

        Returns:
            Figure: 生成的图表对象
        """
        # 默认展示所有可用指标
        if metrics is None:
            # 取第一个结果的所有指标
            first_level = next(iter(sensitivity_results))
            metrics = list(sensitivity_results[first_level].keys())

        # 准备数据
        levels = list(sensitivity_results.keys())
        n_levels = len(levels)
        n_metrics = len(metrics)

        # 创建图表
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 5 * n_metrics))
        if n_metrics == 1:
            axes = [axes]  # 确保axes是列表

        # 为每个指标绘制图表
        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = [sensitivity_results[level][metric] for level in levels]

            # 绘制条形图
            bars = ax.bar(levels, values, color=[self.color_map.get(level, '#1f77b4') for level in levels])

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')

            # 设置标题和标签
            ax.set_title(f'{metric.replace("_", " ").title()}', fontproperties=self.font_properties, fontsize=12)
            ax.set_xlabel('Risk Levels', fontproperties=self.font_properties, fontsize=10)
            ax.set_ylabel('Sensitivity Value', fontproperties=self.font_properties, fontsize=10)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        # 设置总标题
        fig.suptitle(title, fontproperties=self.font_properties, fontsize=14, y=0.98)

        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间

        # 保存图表
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        return fig

    def plot_threshold_impact(self,
                             impact_results: Dict[str, float],
                             title: str = "Risk Threshold Impact Analysis",
                             output_path: Optional[str] = None) -> Figure:
        """绘制阈值影响分析图

        Args:
            impact_results (Dict[str, float]): 阈值影响分析结果
            title (str): 图表标题
            output_path (str, optional): 输出文档路径

        Returns:
            Figure: 生成的图表对象
        """
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 左侧图：类别变化概率
        category_change = impact_results.get('category_change_probability', 0)
        ax1.bar(['Category Change'], [category_change], color='#3498db')
        ax1.bar(['Category Stability'], [1 - category_change], color='#2ecc71', bottom=[category_change])

        # 添加标签
        ax1.text(0, category_change/2, f'{category_change:.1%}', ha='center', va='center', color='white', fontweight='bold')
        ax1.text(0, category_change + (1-category_change)/2, f'{(1-category_change):.1%}', ha='center', va='center', color='white', fontweight='bold')

        # 设置标题和标签
        ax1.set_title('Risk Category Stability', fontproperties=self.font_properties, fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Probability', fontproperties=self.font_properties, fontsize=10)
        ax1.set_yticks(np.arange(0, 1.1, 0.1))
        ax1.set_yticklabels([f'{x:.0%}' for x in np.arange(0, 1.1, 0.1)])

        # 右侧图：风险等级分布
        risk_distribution = impact_results.get('risk_distribution', {})
        levels = list(risk_distribution.keys())
        values = list(risk_distribution.values())

        # 绘制饼图
        if risk_distribution:
            wedges, texts, autotexts = ax2.pie(
                values,
                labels=levels,
                autopct='%1.1f%%',
                startangle=90,
                colors=[self.color_map.get(level, '#1f77b4') for level in levels]
            )

            # 设置标题
            ax2.set_title('Risk Level Distribution', fontproperties=self.font_properties, fontsize=12)

            # 高亮基线类别
            baseline_category = impact_results.get('baseline_category')
            if baseline_category in levels:
                idx = levels.index(baseline_category)
                wedges[idx].set_edgecolor('black')
                wedges[idx].set_linewidth(2)

                # 添加图例
                legend_elements = [Patch(facecolor='none', edgecolor='black', linewidth=2, label='Baseline Category')]
                ax2.legend(handles=legend_elements, loc='lower right')

        # 设置总标题
        fig.suptitle(title, fontproperties=self.font_properties, fontsize=14, y=0.98)

        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间

        # 保存图表
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        return fig

    def plot_sensitivity_radar(self,
                              sensitivity_data: Dict[str, Dict[str, float]],
                              metrics: List[str] = None,
                              title: str = "Factor Sensitivity Radar",
                              output_path: Optional[str] = None,
                              top_n: int = 5) -> Figure:
        """绘制因素敏感度雷达图

        Args:
            sensitivity_data (Dict[str, Dict[str, float]]): 敏感度分析结果
            metrics (List[str], optional): 要展示的指标列表
            title (str): 图表标题
            output_path (str, optional): 输出文档路径
            top_n (int): 展示敏感度最高的前N个因素

        Returns:
            Figure: 生成的图表对象
        """
        # 默认展示 'direct_sensitivity' 和 'relative_influence'
        if metrics is None:
            metrics = ['direct_sensitivity', 'relative_influence']

        # 提取数据并排序
        factors = list(sensitivity_data.keys())

        # 如果数据不足，直接使用所有因素
        if len(factors) <= top_n:
            top_factors = factors
        else:
            # 基于direct_sensitivity排序并取前N个
            sorted_factors = sorted(
                factors,
                key=lambda x: sensitivity_data[x].get('direct_sensitivity', 0),
                reverse=True
            )
            top_factors = sorted_factors[:top_n]

        # 创建雷达图
        fig, ax = plt.subplots(figsize=self.default_figsize, subplot_kw=dict(polar=True))

        # 角度设置
        angles = np.linspace(0, 2*np.pi, len(top_factors), endpoint=False).tolist()
        # 确保闭合
        angles += angles[:1]

        # 绘制每个指标
        for i, metric in enumerate(metrics):
            values = [sensitivity_data[factor].get(metric, 0) for factor in top_factors]
            # 确保闭合
            values += values[:1]

            # 绘制雷达线
            ax.plot(angles, values, linewidth=2, label=metric.replace('_', ' ').title())
            ax.fill(angles, values, alpha=0.1)

        # 添加标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(top_factors, fontproperties=self.font_properties)

        # 设置y轴范围
        ax.set_ylim(0, None)

        # 添加图例和标题
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title(title, fontproperties=self.font_properties, fontsize=14, y=1.08)

        # 设置网格样式
        ax.grid(True, linestyle='--', alpha=0.7)

        # 调整布局
        plt.tight_layout()

        # 保存图表
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        return fig