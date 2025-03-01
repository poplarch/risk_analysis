# risk_visualization.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional, Union, Any
import platform
import logging

# 尝试导入Sankey图所需的库
try:
    from matplotlib.sankey import Sankey
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors
    SANKEY_AVAILABLE = True
except ImportError:
    SANKEY_AVAILABLE = False
    logging.warning("未能导入Sankey图所需的库，将使用替代可视化方法")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiskVisualization:
    """风险可视化类，用于生成风险分析的各种可视化图表"""
    
    def __init__(self, output_dir: str = "output/visualizations",
                 dpi: int = 300,
                 risk_levels: List[str] = None):
        """
        初始化风险可视化类
        
        参数:
            output_dir (str): 输出目录
            dpi (int): 图像分辨率
            risk_levels (List[str]): 风险等级列表
        """
        self.output_dir = output_dir
        self.dpi = dpi
        self.risk_levels = risk_levels or ["VL", "L", "M", "H", "VH"]
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置中文字体
        self.font_properties = self._get_chinese_font()
        self._configure_chinese_fonts()
        
        # 设置风险等级颜色映射
        self.risk_colors = {
            "VL": "#1a9850",  # 绿色
            "L": "#91cf60",    # 浅绿色
            "M": "#ffffbf",    # 黄色
            "H": "#fc8d59",    # 橙色
            "VH": "#d73027"    # 红色
        }
    
    def _get_chinese_font(self) -> FontProperties:
        """
        获取适合当前系统的中文字体
        
        返回:
            FontProperties: 字体属性对象
        """
        system = platform.system()
        
        if system == 'Windows':
            # Windows系统使用微软雅黑或黑体
            fonts = [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf']
            for font_path in fonts:
                if os.path.exists(font_path):
                    return FontProperties(fname=font_path)
        
        elif system == 'Darwin':  # macOS
            # macOS系统使用苹方或华文黑体
            fonts = [
                '/System/Library/Fonts/PingFang.ttc',
                '/Library/Fonts/Hiragino Sans GB.ttc',
                '/System/Library/Fonts/STHeiti Light.ttc'
            ]
            for font_path in fonts:
                if os.path.exists(font_path):
                    return FontProperties(fname=font_path)
        
        # 其他系统使用默认字体
        return FontProperties()
    
    def _configure_chinese_fonts(self) -> None:
        """配置matplotlib支持中文显示"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    def plot_criteria_weights_pie(self, 
                                 weights: Dict[str, float], 
                                 title: str = "一级风险因素局部权重",
                                 filename: str = "criteria_weights_pie.png") -> None:
        """
        绘制一级风险因素局部权重饼状图
        
        参数:
            weights (Dict[str, float]): 权重字典
            title (str): 图表标题
            filename (str): 输出文件名
        """
        # 创建图表
        plt.figure(figsize=(10, 8))
        
        # 提取数据
        labels = list(weights.keys())
        sizes = list(weights.values())
        
        # 计算百分比
        total = sum(sizes)
        percentages = [s/total*100 for s in sizes]
        
        # 设置扇形颜色
        colors = plt.cm.Paired(np.linspace(0, 1, len(labels)))
        
        # 绘制饼图
        patches, texts, autotexts = plt.pie(
            sizes, 
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'fontproperties': self.font_properties}
        )
        
        # 设置属性
        plt.axis('equal')  # 确保饼图是圆形的
        
        # 设置标题
        plt.title(title, fontproperties=self.font_properties, fontsize=16, pad=20)
        
        # 添加图例
        plt.legend(
            patches, 
            [f"{l} ({s:.2%})" for l, s in zip(labels, [s/total for s in sizes])],
            loc="best", 
            bbox_to_anchor=(1, 0.5),
            prop=self.font_properties
        )
        
        # 保存图表
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"饼状图已保存到: {output_path}")
    
    def plot_global_weights_pie(self, 
                              weights: Dict[str, float], 
                              title: str = "二级风险因素全局权重",
                              top_n: int = 10,
                              filename: str = "global_weights_pie.png") -> None:
        """
        绘制二级风险因素全局权重饼状图
        
        参数:
            weights (Dict[str, float]): 权重字典
            title (str): 图表标题
            top_n (int): 显示权重最大的前N个因素
            filename (str): 输出文件名
        """
        # 对权重排序
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        # 选择前N个因素
        if len(sorted_weights) > top_n:
            top_weights = sorted_weights[:top_n]
            # 将剩余权重合并为"其他"
            other_weight = sum(w for _, w in sorted_weights[top_n:])
            if other_weight > 0:
                top_weights.append(("其他", other_weight))
        else:
            top_weights = sorted_weights
        
        # 提取数据
        labels = [w[0] for w in top_weights]
        sizes = [w[1] for w in top_weights]
        
        # 计算百分比
        total = sum(sizes)
        percentages = [s/total*100 for s in sizes]
        
        # 创建图表
        plt.figure(figsize=(12, 9))
        
        # 设置扇形颜色
        colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
        
        # 绘制饼图
        wedges, texts, autotexts = plt.pie(
            sizes, 
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'fontproperties': self.font_properties},
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
        )
        
        # 设置属性
        plt.axis('equal')  # 确保饼图是圆形的
        
        # 设置标题
        plt.title(title, fontproperties=self.font_properties, fontsize=16, pad=20)
        
        # 添加图例
        plt.legend(
            wedges, 
            [f"{l} ({s:.2%})" for l, s in zip(labels, [s/total for s in sizes])],
            loc="center left", 
            bbox_to_anchor=(1, 0.5),
            prop=self.font_properties
        )
        
        # 保存图表
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"全局权重饼状图已保存到: {output_path}")
    
    def plot_fuzzy_membership_bar(self, 
                                 fuzzy_result: np.ndarray,
                                 title: str = "模糊综合评价隶属度",
                                 filename: str = "fuzzy_membership_bar.png") -> None:
        """
        绘制模糊综合评价隶属度柱状图
        
        参数:
            fuzzy_result (np.ndarray): 模糊评价结果向量
            title (str): 图表标题
            filename (str): 输出文件名
        """
        # 创建图表
        plt.figure(figsize=(10, 6))
        
        # 绘制柱状图
        bars = plt.bar(
            self.risk_levels, 
            fuzzy_result,
            color=[self.risk_colors.get(level, '#1f77b4') for level in self.risk_levels]
        )
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.01,
                f'{height:.3f}',
                ha='center', 
                va='bottom'
            )
        
        # 设置坐标轴标签
        plt.xlabel('风险等级', fontproperties=self.font_properties, fontsize=12)
        plt.ylabel('隶属度', fontproperties=self.font_properties, fontsize=12)
        
        # 设置标题
        plt.title(title, fontproperties=self.font_properties, fontsize=14)
        
        # 添加网格线
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 设置坐标轴范围
        plt.ylim(0, max(fuzzy_result) * 1.2)
        
        # 保存图表
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"模糊隶属度柱状图已保存到: {output_path}")
    
    def plot_sensitivity_radar(self, 
                              sensitivity_indices: Dict[str, float],
                              title: str = "风险因素敏感性雷达图",
                              top_n: int = 8,
                              filename: str = "sensitivity_radar.png") -> None:
        """
        绘制敏感性雷达图
        
        参数:
            sensitivity_indices (Dict[str, float]): 敏感性指标字典
            title (str): 图表标题
            top_n (int): 显示敏感性最大的前N个因素
            filename (str): 输出文件名
        """
        # 对敏感性指标排序
        sorted_indices = sorted(
            sensitivity_indices.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        # 选择前N个敏感性最大的因素
        if len(sorted_indices) > top_n:
            selected_indices = sorted_indices[:top_n]
        else:
            selected_indices = sorted_indices
        
        # 提取数据
        factors = [item[0] for item in selected_indices]
        values = [abs(item[1]) for item in selected_indices]  # 使用绝对值
        
        # 确保数据点数量足够
        if len(factors) < 3:
            logger.warning("雷达图需要至少3个数据点，当前只有%d个", len(factors))
            return
        
        # 创建图表
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # 计算角度
        angles = np.linspace(0, 2*np.pi, len(factors), endpoint=False).tolist()
        
        # 闭合雷达图
        values.append(values[0])
        angles.append(angles[0])
        factors.append(factors[0])
        
        # 绘制雷达图
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        
        # 设置标签
        ax.set_thetagrids(
            np.degrees(angles[:-1]), 
            factors[:-1],
            fontproperties=self.font_properties
        )
        
        # 设置标题
        ax.set_title(title, fontproperties=self.font_properties, fontsize=14, y=1.1)
        
        # 添加网格线
        ax.grid(True)
        
        # 保存图表
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"敏感性雷达图已保存到: {output_path}")
    
    def plot_risk_level_sankey(self, 
                              risk_distribution: Dict[str, float],
                              baseline_category: str,
                              title: str = "风险等级变化流向图",
                              filename: str = "risk_level_sankey.png") -> None:
        """
        绘制风险等级变化Sankey图
        
        参数:
            risk_distribution (Dict[str, float]): 风险等级分布概率
            baseline_category (str): 基准风险等级
            title (str): 图表标题
            filename (str): 输出文件名
        """
        if not SANKEY_AVAILABLE:
            # 如果Sankey库不可用，使用替代可视化方法
            self._plot_risk_level_alternative(
                risk_distribution, 
                baseline_category, 
                title, 
                filename
            )
            return
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 初始化Sankey图
        sankey = Sankey(ax=plt.gca(), scale=0.01, offset=0.2, head_angle=120)
        
        # 准备Sankey图的流数据
        flows = []
        for level, probability in risk_distribution.items():
            if level != baseline_category and probability > 0.01:  # 忽略很小的概率
                flows.append((baseline_category, level, probability))
        
        # 添加流
        baseline_sum = sum(v for k, v in risk_distribution.items() if k != baseline_category)
        if baseline_sum < 1.0:
            # 添加保持在基准等级的流
            flows.append((baseline_category, baseline_category, 1.0 - baseline_sum))
        
        # 配置Sankey图
        for source, target, value in flows:
            sankey.add(
                flows=[value],
                orientations=[0],
                labels=[source, target],
                trunklength=10.0,
                pathlengths=[5],
                patchlabel=None
            )
        
        # 绘制Sankey图
        diagrams = sankey.finish()
        
        # 设置标题
        plt.title(title, fontproperties=self.font_properties, fontsize=14)
        
        # 保存图表
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"风险等级流向图已保存到: {output_path}")
    
    def _plot_risk_level_alternative(self, 
                                    risk_distribution: Dict[str, float],
                                    baseline_category: str,
                                    title: str = "风险等级分布",
                                    filename: str = "risk_level_distribution.png") -> None:
        """
        绘制风险等级分布替代图（当Sankey图不可用时）
        
        参数:
            risk_distribution (Dict[str, float]): 风险等级分布概率
            baseline_category (str): 基准风险等级
            title (str): 图表标题
            filename (str): 输出文件名
        """
        # 创建图表
        plt.figure(figsize=(10, 6))
        
        # 提取数据
        levels = list(risk_distribution.keys())
        values = list(risk_distribution.values())
        
        # 设置颜色
        colors = [
            self.risk_colors.get(level, '#1f77b4') 
            if level != baseline_category else '#1f77b4'
            for level in levels
        ]
        
        # 突出显示基准等级
        explode = [0.1 if level == baseline_category else 0 for level in levels]
        
        # 绘制饼图
        wedges, texts, autotexts = plt.pie(
            values, 
            labels=levels,
            explode=explode,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'fontproperties': self.font_properties},
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
        )
        
        # 设置属性
        plt.axis('equal')
        
        # 设置标题
        plt.title(title, fontproperties=self.font_properties, fontsize=14)
        
        # 添加基准等级标记
        plt.annotate(
            f"基准等级: {baseline_category}",
            xy=(0, 0),
            xytext=(0, -0.2),
            fontproperties=self.font_properties,
            fontsize=12,
            ha='center'
        )
        
        # 保存图表
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"风险等级分布图已保存到: {output_path}")
    
    def plot_sensitivity_tornado(self, 
                               sensitivity_indices: Dict[str, float],
                               title: str = "风险因素敏感性龙卷风图",
                               top_n: int = 10,
                               filename: str = "sensitivity_tornado.png") -> None:
        """
        绘制敏感性龙卷风图
        
        参数:
            sensitivity_indices (Dict[str, float]): 敏感性指标字典
            title (str): 图表标题
            top_n (int): 显示敏感性最大的前N个因素
            filename (str): 输出文件名
        """
        # 对敏感性指标排序
        sorted_indices = sorted(
            sensitivity_indices.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        # 选择前N个敏感性最大的因素
        if len(sorted_indices) > top_n:
            selected_indices = sorted_indices[:top_n]
        else:
            selected_indices = sorted_indices
        
        # 提取数据
        factors = [item[0] for item in selected_indices]
        values = [item[1] for item in selected_indices]
        
        # 创建图表
        plt.figure(figsize=(10, 8))
        
        # 绘制水平条形图
        bars = plt.barh(
            factors,
            values,
            color=plt.cm.coolwarm(
                np.array([0.5 + v/10 for v in values])
            ),
            height=0.6
        )
        
        # 添加垂直参考线
        plt.axvline(x=0, color='gray', linestyle='-', linewidth=1)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ha = 'left' if width < 0 else 'right'
            x = width + 0.01 if width > 0 else width - 0.01
            plt.text(
                x, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                ha=ha, 
                va='center',
                color='black'
            )
        
        # 设置坐标轴标签
        plt.xlabel('敏感性指标', fontproperties=self.font_properties, fontsize=12)
        plt.ylabel('风险因素', fontproperties=self.font_properties, fontsize=12)
        
        # 设置刻度标签字体
        plt.xticks(fontproperties=self.font_properties)
        plt.yticks(fontproperties=self.font_properties)
        
        # 添加网格线
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 设置标题
        plt.title(title, fontproperties=self.font_properties, fontsize=14)
        
        # 保存图表
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"敏感性龙卷风图已保存到: {output_path}")
    
    def plot_risk_heatmap(self, 
                         risk_matrix: np.ndarray,
                         factors: List[str],
                         variations: List[float],
                         title: str = "风险影响热力图",
                         filename: str = "risk_heatmap.png") -> None:
        """
        绘制风险影响热力图
        
        参数:
            risk_matrix (np.ndarray): 风险指数矩阵
            factors (List[str]): 风险因素名称
            variations (List[float]): 变化率列表
            title (str): 图表标题
            filename (str): 输出文件名
        """
        # 创建图表
        plt.figure(figsize=(10, 8))
        
        # 设置热力图标签
        x_labels = [f"{v*100:.0f}%" for v in variations]
        y_labels = [f"{v*100:.0f}%" for v in variations]
        
        # 绘制热力图
        im = plt.imshow(risk_matrix, cmap="YlOrRd")
        
        # 添加颜色条
        cbar = plt.colorbar(im)
        cbar.set_label('风险指数', fontproperties=self.font_properties, fontsize=12)
        
        # 添加标签
        plt.xlabel(f'{factors[1]} 变化率', fontproperties=self.font_properties, fontsize=12)
        plt.ylabel(f'{factors[0]} 变化率', fontproperties=self.font_properties, fontsize=12)
        
        # 设置刻度标签
        plt.xticks(range(len(x_labels)), x_labels, fontproperties=self.font_properties)
        plt.yticks(range(len(y_labels)), y_labels, fontproperties=self.font_properties)
        
        # 添加值标注
        for i in range(risk_matrix.shape[0]):
            for j in range(risk_matrix.shape[1]):
                plt.text(
                    j, i, f"{risk_matrix[i, j]:.3f}",
                    ha="center", va="center",
                    color="black" if risk_matrix[i, j] < 0.7 else "white"
                )
        
        # 设置标题
        plt.title(title, fontproperties=self.font_properties, fontsize=14)
        
        # 保存图表
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"风险热力图已保存到: {output_path}")
    
    def plot_monte_carlo_scatter(self, 
                               risk_indices: List[float],
                               weights: List[Dict[str, float]],
                               title: str = "蒙特卡洛风险模拟",
                               top_factors: int = 2,
                               filename: str = "monte_carlo_scatter.png") -> None:
        """
        绘制蒙特卡洛模拟散点图
        
        参数:
            risk_indices (List[float]): 风险指数列表
            weights (List[Dict[str, float]]): 模拟权重列表
            title (str): 图表标题
            top_factors (int): 显示相关性最高的前N个因素
            filename (str): 输出文件名
        """
        # 检查数据有效性
        if not risk_indices or not weights:
            logger.warning("蒙特卡洛模拟数据为空，无法生成散点图")
            return
        
        # 选择与风险指数相关性最高的因素
        correlations = {}
        for factor in weights[0].keys():
            factor_weights = [w.get(factor, 0) for w in weights]
            if len(set(factor_weights)) > 1:  # 确保因素权重有变化
                corr = np.corrcoef(factor_weights, risk_indices)[0, 1]
                correlations[factor] = abs(corr)  # 使用相关系数的绝对值
        
        # 对相关性排序
        sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # 选择相关性最高的N个因素
        selected_factors = [item[0] for item in sorted_corr[:top_factors]]
        
        # 如果因素不足，使用可用的所有因素
        if len(selected_factors) < top_factors:
            selected_factors = list(correlations.keys())
        
        # 创建散点图矩阵
        fig, axes = plt.subplots(
            nrows=len(selected_factors),
            ncols=1,
            figsize=(10, 4 * len(selected_factors)),
            sharex=True
        )
        
        # 如果只有一个因素，确保axes是列表
        if len(selected_factors) == 1:
            axes = [axes]
        
        # 为每个选定的因素绘制散点图
        for i, factor in enumerate(selected_factors):
            ax = axes[i]
            
            # 提取数据
            factor_weights = [w.get(factor, 0) for w in weights]
            
            # 绘制散点图
            scatter = ax.scatter(
                factor_weights,
                risk_indices,
                c=risk_indices,
                cmap='viridis',
                alpha=0.6,
                edgecolors='w',
                linewidth=0.5
            )
            
            # 添加趋势线
            z = np.polyfit(factor_weights, risk_indices, 1)
            p = np.poly1d(z)
            ax.plot(
                factor_weights,
                p(factor_weights),
                "r--",
                alpha=0.8
            )
            
            # 添加相关性信息
            corr = np.corrcoef(factor_weights, risk_indices)[0, 1]
            ax.text(
                0.05, 0.95,
                f"相关系数: {corr:.3f}",
                transform=ax.transAxes,
                fontproperties=self.font_properties,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )
            
            # 设置标签
            ax.set_ylabel('风险指数', fontproperties=self.font_properties, fontsize=12)
            ax.set_title(
                f"因素: {factor}",
                fontproperties=self.font_properties,
                fontsize=12
            )
            
            # 添加网格线
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # 为最后一个子图设置x轴标签
        axes[-1].set_xlabel('因素权重', fontproperties=self.font_properties, fontsize=12)
        
        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=axes.ravel().tolist())
        cbar.set_label('风险指数', fontproperties=self.font_properties, fontsize=12)
        
        # 设置总标题
        fig.suptitle(title, fontproperties=self.font_properties, fontsize=16)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
        
        # 保存图表
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"蒙特卡洛散点图已保存到: {output_path}")
    
    def plot_risk_distribution(self, 
                              risk_indices: List[float],
                              risk_stats: Dict[str, Any],
                              title: str = "风险指数分布",
                              filename: str = "risk_distribution.png") -> None:
        """
        绘制风险指数分布图
        
        参数:
            risk_indices (List[float]): 风险指数列表
            risk_stats (Dict[str, Any]): 风险统计信息
            title (str): 图表标题
            filename (str): 输出文件名
        """
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # 绘制直方图和核密度估计
        sns.histplot(risk_indices, kde=True, ax=ax1, color='skyblue')
        
        # 添加统计信息参考线
        mean = risk_stats.get("mean", np.mean(risk_indices))
        median = risk_stats.get("median", np.median(risk_indices))
        
        ax1.axvline(mean, color='red', linestyle='-', label=f'均值: {mean:.3f}')
        ax1.axvline(median, color='green', linestyle='--', label=f'中位数: {median:.3f}')
        
        # 添加百分位线
        percentiles = risk_stats.get("percentiles", {})
        if percentiles:
            for p_label, p_value in percentiles.items():
                ax1.axvline(
                    p_value,
                    color='orange',
                    linestyle=':',
                    alpha=0.7,
                    label=f'{p_label}: {p_value:.3f}'
                )
        
        # 设置标签
        ax1.set_xlabel('风险指数', fontproperties=self.font_properties, fontsize=12)
        ax1.set_ylabel('频率', fontproperties=self.font_properties, fontsize=12)
        ax1.set_title(title, fontproperties=self.font_properties, fontsize=14)
        ax1.legend(prop=self.font_properties)
        
        # 添加网格线
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制箱线图
        sns.boxplot(x=risk_indices, ax=ax2, color='lightblue')
        
        # 设置标签
        ax2.set_xlabel('风险指数', fontproperties=self.font_properties, fontsize=12)
        ax2.set_title('风险指数箱线图', fontproperties=self.font_properties, fontsize=12)
        
        # 添加统计信息注释
        stats_text = (
            f"均值: {mean:.3f}\n"
            f"中位数: {median:.3f}\n"
            f"标准差: {risk_stats.get('std', np.std(risk_indices)):.3f}\n"
            f"最小值: {risk_stats.get('min', np.min(risk_indices)):.3f}\n"
            f"最大值: {risk_stats.get('max', np.max(risk_indices)):.3f}"
        )
        
        ax2.text(
            0.95, 0.50,
            stats_text,
            transform=ax2.transAxes,
            fontproperties=self.font_properties,
            fontsize=10,
            va='center',
            ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"风险分布图已保存到: {output_path}")
