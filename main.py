# -*- coding: utf-8 -*-
# TODO: 层次分析法支持4层模型，包括目标层、准则层、指标层、方案层（可选）
# TODO: 增加GUI支持、多层次模糊综合评价法
"""
风险分析工具主程序
支持AHP和增强型FCE方法的集成风险分析系统
"""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

# 导入自定义模块
from ahp_processor import AHPProcessor
from enhanced_fuzzy_evaluator import EnhancedFuzzyEvaluator
from excel_handler import ExcelDataHandler, ExcelExporter
from risk_sensitivity_analyzer import RiskSensitivityAnalyzer
from risk_visualization import RiskVisualization


# 配置日志
def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO) -> None:
    """
    设置日志配置

    Args:
        log_dir: 日志目录
        log_level: 日志级别
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/risk_analysis_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )

    # 设置第三方库的日志级别
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

def create_default_config() -> Dict:
    """
    Create default configuration dictionary with sensible defaults

    Returns:
        Dict: Default configuration structure
    """
    # Default configuration structure
    default_config = {
        "project": {
            "name": "信创项目风险分析",
            "description": "商业银行信息系统信创项目风险分析",
            "version": "1.0.0"
        },
        "paths": {
            "input_dir": "input/",
            "output_dir": "output/",
            "log_dir": "logs/"
        },
        "files": {
            "ahp_excel_prefix": "ahp_template_",
            "ahp_results_prefix": "ahp_results_",
            "fuzzy_excel": "fuzzy_risk_evaluation.xlsx"
        },
        "ahp_settings": {
            "ahp_model_level": ["Goal", "Criteria1", "Criteria2", "Criteria3", "Criteria4", "Criteria5"],
            "correction_method": "LLSM",
            "aggregation_method": "geometric",
            "weight_method": "eigenvector",
            "expert_weights_enabled": True,
        },
        "fuzzy_settings": {
            "use_dynamic_membership": True,
            "weight_threshold": 0.03,
            "risk_levels": ["VL", "L", "M", "H", "VH"]
        },
        "analysis_options": {
            "perform_sensitivity_analysis": True,
            "sensitivity_depth": 2,
            "cross_sensitivity_enabled": True
        },
        "visualization": {
            "enabled": True,
            "output_formats": ["png"],
            "dpi": 300,
            "color_scheme": "viridis"
        }
    }

    return default_config


def validate_config(config: Dict) -> Dict:
    """
    Validate configuration dictionary and ensure required parameters exist

    Args:
        config: Configuration dictionary to validate

    Returns:
        Dict: Validated and normalized configuration

    Raises:
        FileNotFoundError: If required input files are missing
        ValueError: If configuration parameters are invalid
    """
    # Normalize and validate paths
    for path_key in ["input_dir", "output_dir", "log_dir"]:
        if path_key in config.get("paths", {}):
            # Ensure paths end with "/"
            if not config["paths"][path_key].endswith("/"):
                config["paths"][path_key] += "/"

            # Create directories if they don't exist
            if path_key != "input_dir":  # Don't create input dir
                os.makedirs(config["paths"][path_key], exist_ok=True)
                logging.info(f"Created directory: {config['paths'][path_key]}")

    # Extract key configuration parameters for validation
    input_dir = config.get("paths", {}).get("input_dir", "input/")
    ahp_prefix = config.get("files", {}).get("ahp_excel_prefix", "ahp_template_")
    fuzzy_excel = config.get("files", {}).get("fuzzy_excel", "fuzzy_risk_evaluation.xlsx")
    ahp_model_level = config.get("ahp_settings", {}).get("ahp_model_level")

    # Validate AHP template files existence
    missing_ahp_files = []
    for level in ahp_model_level:
        ahp_file = f"{input_dir}{ahp_prefix}{level}.xlsx"
        if not os.path.exists(ahp_file):
            missing_ahp_files.append(ahp_file)

    if missing_ahp_files:
        error_msg = f"Missing required AHP template files: {', '.join(missing_ahp_files)}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Validate fuzzy evaluation file existence
    fuzzy_file_path = f"{input_dir}{fuzzy_excel}"
    if not os.path.exists(fuzzy_file_path):
        error_msg = f"Missing required fuzzy evaluation file: {fuzzy_file_path}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Validate AHP settings
    valid_correction_methods = ["LLSM", "direct", "iterative"]
    correction_method = config.get("ahp_settings", {}).get("correction_method", "LLSM")
    if correction_method not in valid_correction_methods:
        logging.warning(f"Invalid correction method: {correction_method}. Using default: LLSM")
        config["ahp_settings"]["correction_method"] = "LLSM"

    valid_aggregation_methods = ["geometric", "arithmetic", "weighted"]
    aggregation_method = config.get("ahp_settings", {}).get("aggregation_method", "geometric")
    if aggregation_method not in valid_aggregation_methods:
        logging.warning(f"Invalid aggregation method: {aggregation_method}. Using default: geometric")
        config["ahp_settings"]["aggregation_method"] = "geometric"

    valid_weight_methods = ["eigenvector", "geometric", "arithmetic"]
    weight_method = config.get("ahp_settings", {}).get("weight_method", "eigenvector")
    if weight_method not in valid_weight_methods:
        logging.warning(f"Invalid weight method: {weight_method}. Using default: eigenvector")
        config["ahp_settings"]["weight_method"] = "eigenvector"

    # Validate fuzzy settings
    weight_threshold = config.get("fuzzy_settings", {}).get("weight_threshold", 0.03)
    if not isinstance(weight_threshold, (int, float)) or weight_threshold < 0 or weight_threshold > 1:
        logging.warning(f"Invalid weight threshold: {weight_threshold}. Using default: 0.03")
        config["fuzzy_settings"]["weight_threshold"] = 0.03

    # Validate visualization settings
    dpi = config.get("visualization", {}).get("dpi", 300)
    if not isinstance(dpi, int) or dpi < 72:
        logging.warning(f"Invalid DPI value: {dpi}. Using default: 300")
        config["visualization"]["dpi"] = 300

    # Apply default values for any missing parameters
    default_config = create_default_config()

    # Recursively merge default values for missing keys
    def merge_defaults(target, source):
        for key, value in source.items():
            if key not in target:
                target[key] = value
            elif isinstance(value, dict) and isinstance(target[key], dict):
                merge_defaults(target[key], value)

    merge_defaults(config, default_config)

    logging.info("Configuration validated successfully")
    return config

def load_config(config_path: str) -> Dict:
    """
    Load and validate configuration

    Args:
        config_path: Path to configuration file

    Returns:
        Validated configuration dictionary
    """
    # Load configuration
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from: {config_path}")
    else:
        # Create default configuration
        config = create_default_config()
        os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logging.info(f"Created default configuration: {config_path}")

    # Validate and normalize configuration
    config = validate_config(config)
    return config

def present_hierarchical_weights(
        criteria_weights: Dict[str, float],
        sub_criteria_weights: Dict[str, Dict[str, float]],
        global_weights: Dict[str, float]
) -> None:
    """
    Generates tabular presentation of multi-level weight hierarchies

    Args:
        criteria_weights: Top-level criteria weights
        sub_criteria_weights: Nested dictionary of sub-criteria weights
        global_weights: Calculated global weights
    """
    # Prepare structured presentation of hierarchical data
    print("\n========== Weight Analysis Results ==========")

    # Present primary criteria
    print("\nPrimary Criteria Weights:")
    weights_table = [(k, f"{v:.4f}") for k, v in criteria_weights.items()]
    print(tabulate(
        weights_table,
        headers=["Criterion", "Weight"],
        tablefmt="grid",
        colalign=("left", "right")
    ))

    # Present sub-criteria with local weights
    print("\nSub-Criteria Local Weights:")
    for main_criterion, weights in sub_criteria_weights.items():
        print(f"\n{main_criterion}:")
        local_table = [(k, f"{v:.4f}") for k, v in weights.items()]
        print(tabulate(
            local_table,
            headers=["Sub-Criterion", "Local Weight"],
            tablefmt="grid",
            colalign=("left", "right")
        ))

    # Present global weights (descending order)
    print("\nGlobal Criteria Weights (Descending):")
    global_table = [(k, f"{v:.4f}") for k, v in sorted(
        global_weights.items(),
        key=lambda x: x[1],
        reverse=True
    )]
    print(tabulate(
        global_table,
        headers=["Criterion", "Global Weight"],
        tablefmt="grid",
        colalign=("left", "right")
    ))


def display_risk_analysis_matrix(
        sensitivity_results: pd.DataFrame,
        include_rankings: bool = True
) -> None:
    """
    Presents sensitivity analysis results with comprehensive alignment specifications

    Args:
        sensitivity_results: DataFrame containing sensitivity metrics
        include_rankings: Flag to include ordinal rankings in output
    """
    # Extract and prepare data
    if include_rankings:
        # Add ranking column to input DataFrame
        sensitivity_results['Rank'] = sensitivity_results['敏感性指标'].rank(ascending=False, method='min').astype(int)

    # Define column alignment specifications
    headers = sensitivity_results.columns.tolist()
    alignment_map = {
        '风险因素': 'left',  # Left-align textual identifiers
        '敏感性指标': 'decimal',  # Align numerical values at decimal point
        'Rank': 'center',  # Center-align ordinal rankings
        '原始权重': 'decimal'  # Align numerical values at decimal point
    }

    # Generate column alignment specification
    colalign = [alignment_map.get(col, 'center') for col in headers]

    # Configure presentation parameters
    print(tabulate(
        sensitivity_results,
        headers=headers,
        tablefmt="grid",
        colalign=colalign,
        floatfmt=".4f",  # Standardize floating-point representation
        showindex=False  # Suppress dataframe index display
    ))
def process_ahp_level(
        ahp_processor: AHPProcessor,
        level: str,
        criteria_weights: Dict[str, float],
        sub_criteria_weights: Dict[str, Dict[str, float]],
        excel_results_path: str
) -> None:
    """
    处理单个AHP层级并更新权重字典

    Args:
        ahp_processor: AHP处理器
        level: 层级名称
        criteria_weights: 一级准则权重字典
        sub_criteria_weights: 二级准则权重字典
        excel_results_path: AHP结果文件
    """
    try:
        logging.info(f"处理AHP层级: {level}")
        results = ahp_processor.process_level(level)

        # 更新权重字典
        if level == "Goal":
            # 一级目标层结果更新到criteria_weights
            criteria_weights.update(results["final_weights"])
        else:
            # 子层级结果更新到sub_criteria_weights
            main_criterion = level
            if main_criterion not in sub_criteria_weights:
                sub_criteria_weights[main_criterion] = {}
            sub_criteria_weights[main_criterion].update(results["final_weights"])

        # 打印一致性结果
        print(f"\n层级 {level} 一致性分析:")
        for result in results["correction_results"]:
            status = "已修正" if result.adjusted else "未修正"
            print(f"专家{result.expert_id}: 原始CR={result.original_cr:.4f}, "
                  f"最终CR={result.final_cr:.4f}, 状态: {status}")

        # 打印最终权重
        weights_table = [(k, f"{v:.4f}") for k, v in results["final_weights"].items()]
        print("\n最终权重:")
        print(tabulate(weights_table, headers=["指标", "权重"], tablefmt="grid"))

        # 记录聚合一致性比率
        cr = results["aggregated_consistency"].consistency_ratio
        print(f"聚合矩阵一致性比率(CR): {cr:.4f}")

        # 导出结果
        ExcelExporter().export_ahp_results(results, excel_results_path)
        logging.info(f"已导出AHP结果到: {excel_results_path}")

    except Exception as e:
        logging.error(f"处理层级 {level} 出错: {str(e)}")
        print(f"处理层级 {level} 出错: {str(e)}")


def calculate_global_weights(
        criteria_weights: Dict[str, float],
        sub_criteria_weights: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    计算全局权重

    Args:
        criteria_weights: 一级准则权重
        sub_criteria_weights: 二级准则权重

    Returns:
        全局权重字典
    """
    global_weights = {}

    # 计算全局权重 = 一级权重 * 二级权重
    for criteria, local_weights in sub_criteria_weights.items():
        if criteria in criteria_weights:
            for sub_criteria, local_weight in local_weights.items():
                global_weights[sub_criteria] = local_weight * criteria_weights[criteria]
        else:
            logging.warning(f"一级准则 {criteria} 在权重字典中未找到")
            for sub_criteria, local_weight in local_weights.items():
                global_weights[sub_criteria] = local_weight * 0  # 设为0权重

    return global_weights


def perform_enhanced_fuzzy_evaluation(
        global_weights: Dict[str, float],
        excel_handler: ExcelDataHandler,
        fuzzy_config: Dict
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    执行增强型模糊综合评价

    Args:
        global_weights: 全局权重字典
        excel_handler: Excel数据处理器
        fuzzy_config: 模糊综合评价配置信息

    Returns:
        评价结果字典
        参与模糊综合评价的二级风险因素全局权重字典
    """
    try:
        # 初始化增强型模糊评价器
        fuzzy_evaluator = EnhancedFuzzyEvaluator(
            risk_levels = fuzzy_config["risk_levels"],
            dynamic_enabled = fuzzy_config["use_dynamic_membership"]
        )

        # 读取专家评分数据
        scores_df, fce_expert_weights = excel_handler.read_expert_scores(fuzzy_config["fuzzy_excel"])
        if fce_expert_weights is None:
            fce_expert_weights = [1.0/scores_df.shape[1]] * scores_df.shape[1]

        # 过滤权重大于阈值的风险因素
        weight_threshold = fuzzy_config.get("weight_threshold", 0.03)
        significant_factors = {k: v for k, v in global_weights.items() if v >= weight_threshold}

        if not significant_factors:
            logging.warning(f"没有权重大于{weight_threshold}的风险因素，使用全部因素")
            significant_factors = global_weights

        # 准备评价数据
        expert_scores = {}
        for factor in scores_df.index:
            if factor in significant_factors:
                expert_scores[factor] = scores_df.loc[factor].values

        logging.info(f"执行模糊综合评价，使用{len(expert_scores)}个风险因素")

        # 执行模糊综合评价
        evaluation_results = fuzzy_evaluator.evaluate(
            expert_scores=expert_scores,
            factor_weights=significant_factors,
            expert_weights=fce_expert_weights,
        )

        # 如果需要进行敏感性分析
        if fuzzy_config["perform_sensitivity_analysis"]:
            logging.info("执行敏感性分析...")

            # 单因素敏感性分析
            sensitivity_results = fuzzy_evaluator.perform_sensitivity_analysis(
                factor_weights=significant_factors,
                expert_scores=expert_scores,
                expert_weights=fce_expert_weights,
            )
            #sensitivity_results = fuzzy_evaluator.enhanced_sensitivity_analysis(
            #    factor_weights=significant_factors,
            #    expert_scores=expert_scores,
            #    expert_weights=fce_expert_weights,
            #)

            ## 识别最敏感的两个因素进行交叉敏感性分析
            top_factors = sensitivity_results["ranked_factors"][:2]
            if len(top_factors) == 2:
                cross_results = fuzzy_evaluator.cross_sensitivity_analysis(
                    factor_weights=significant_factors,
                    expert_scores=expert_scores,
                    factors=top_factors,
                    expert_weights=fce_expert_weights,
                )
            else:
                cross_results = None
                logging.warning("敏感因素不足，无法进行交叉敏感性分析")

            # 合并敏感性分析结果
            evaluation_results["sensitivity_analysis"] = sensitivity_results
            if cross_results:
                evaluation_results["cross_sensitivity"] = cross_results

        return evaluation_results, significant_factors

    except Exception as e:
        logging.error(f"模糊综合评价出错: {str(e)}")
        raise


def visualize_results_(criteria_weights: Dict[str, Any], global_weights: Dict[str, Any], fuzzy_result: np.array) -> None:
    # 创建敏感性分析器
    analyzer = RiskSensitivityAnalyzer(global_weights, fuzzy_result)

    # 执行单因素敏感性分析
    single_factor_results = analyzer.single_factor_sensitivity()

    top_factors = single_factor_results["ranked_factors"][:2]
    # 执行交叉敏感性分析
    cross_factor_results = analyzer.cross_factor_sensitivity(
        factors=top_factors
    )

    # 执行蒙特卡洛敏感性分析
    #monte_carlo_results = analyzer.monte_carlo_sensitivity(num_simulations=500)

    # 计算风险阈值影响
    #threshold_impact = analyzer.calculate_threshold_impact()

    # 创建可视化器
    visualizer = RiskVisualization(output_dir="output/visualizations")

    # 绘制一级风险因素局部权重饼状图
    # 首先，提取一级风险因素的权重
    visualizer.plot_criteria_weights_pie(criteria_weights)

    # 绘制二级风险因素全局权重饼状图
    visualizer.plot_global_weights_pie(global_weights)

    # 绘制模糊综合评价隶属度柱状图
    visualizer.plot_fuzzy_membership_bar(fuzzy_result)

    # 绘制敏感性雷达图
    visualizer.plot_sensitivity_radar(single_factor_results["sensitivity_indices"])

    # 绘制敏感性Tornado图
    #visualizer.plot_sensitivity_tornado(single_factor_results["sensitivity_indices"])

    # 绘制风险影响热力图
    visualizer.plot_risk_heatmap(
        cross_factor_results["risk_matrix"],
        cross_factor_results["factors"],
        cross_factor_results["variations"]
    )

    # 绘制风险等级变化Sankey图
    #visualizer.plot_risk_level_sankey(
    #    threshold_impact["risk_distribution"],
    #    threshold_impact["baseline_category"]
    #)

    #visualizer.plot_risk_level_transition(
    #    threshold_impact["risk_distribution"],
    #    threshold_impact["baseline_category"]
    #)

    #visualizer.plot_risk_level_network(
    #    threshold_impact["risk_distribution"],
    #    threshold_impact["baseline_category"]
    #)


    # 绘制蒙特卡洛模拟散点图
    #visualizer.plot_monte_carlo_scatter(
    #    monte_carlo_results["risk_indices"],
    #    monte_carlo_results["simulation_weights"]
    #)

    # 绘制风险分布图
    #visualizer.plot_risk_distribution(
    #    monte_carlo_results["risk_indices"],
    #    monte_carlo_results["risk_stats"]
    #)

    print("所有图表已生成!")

def visualize_results(
        evaluation_results: Dict[str, Any],
        config: Dict,
        output_dir: str
) -> None:
    """
    可视化分析结果

    参数:
        evaluation_results: 评价结果
        config: 配置信息
        output_dir: 输出目录
    """
    if not config.get("visualization", {}).get("enabled", True):
        logging.info("可视化功能已禁用")
        return

    try:
        # 创建可视化目录
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # 初始化模糊评价器
        fuzzy_evaluator = EnhancedFuzzyEvaluator(
            dynamic_enabled=config.get("fuzzy_settings", {}).get("use_dynamic_membership", True)
        )

        # 获取中文字体属性 - 直接使用实例中的字体属性
        font_prop = fuzzy_evaluator.font_properties

        # 1. 可视化隶属度函数
        if "factor_membership" in evaluation_results:
            # 获取第一个因素的评分样本用于演示
            sample_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # 示例评分点

            fuzzy_evaluator.visualize_membership_functions(
                use_dynamic=config.get("fuzzy_settings", {}).get("use_dynamic_membership", True),
                scores=sample_scores,
                output_path=os.path.join(viz_dir, "membership_functions.png")
            )
            logging.info("已生成隶属度函数可视化")

        # 2. 可视化模糊评价结果
        if "integrated_result" in evaluation_results:
            result = evaluation_results["integrated_result"]

            plt.figure(figsize=(10, 6))
            bars = plt.bar(["VL", "L", "M", "H", "VH"], result)

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{height:.3f}', ha='center', va='bottom')

            # 使用中文字体
            plt.title("风险等级隶属度分布", fontproperties=font_prop, fontsize=14)
            plt.ylabel("隶属度", fontproperties=font_prop, fontsize=12)
            plt.ylim(0, max(result) * 1.2)
            plt.grid(True, alpha=0.3)

            plt.savefig(os.path.join(viz_dir, "fuzzy_results.png"), dpi=300, bbox_inches='tight')
            plt.close()
            logging.info("已生成模糊评价结果可视化")

        # 3. 可视化敏感性分析结果
        if "sensitivity_analysis" in evaluation_results:
            fuzzy_evaluator.visualize_sensitivity_analysis(
                sensitivity_results=evaluation_results["sensitivity_analysis"],
                top_n=8,  # 显示前8个最敏感的因素
                output_dir=viz_dir
            )
            logging.info("已生成敏感性分析可视化")

        # 4. 可视化交叉敏感性分析结果
        if "cross_sensitivity" in evaluation_results:
            fuzzy_evaluator.visualize_cross_sensitivity(
                cross_results=evaluation_results["cross_sensitivity"],
                output_dir=viz_dir
            )
            logging.info("已生成交叉敏感性分析可视化")

    except Exception as e:
        logging.error(f"生成可视化图表出错: {str(e)}", exc_info=True)
        print(f"生成可视化图表出错: {str(e)}")

def export_evaluation_results(
        evaluation_results: Dict[str, Any],
        output_dir: str
) -> None:
    """
    导出评价结果到Excel

    Args:
        evaluation_results: 评价结果
        output_dir: 输出目录
    """
    try:
        # 创建时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"fuzzy_evaluation_results_{timestamp}.xlsx")

        # 创建Excel写入器
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 1. 导出风险等级隶属度
            if "integrated_result" in evaluation_results:
                result = evaluation_results["integrated_result"]
                risk_df = pd.DataFrame([result], columns=["VL", "L", "M", "H", "VH"],
                                       index=["风险等级隶属度"])
                risk_df.to_excel(writer, sheet_name="风险等级隶属度")

            # 2. 导出风险指数
            if "risk_index" in evaluation_results:
                risk_index = evaluation_results["risk_index"]
                index_df = pd.DataFrame([{"风险指数": risk_index}])
                index_df.to_excel(writer, sheet_name="风险指数", index=False)

            # 3. 导出各因素隶属度
            if "factor_membership" in evaluation_results:
                factor_membership = evaluation_results["factor_membership"]
                membership_data = []

                for factor, membership in factor_membership.items():
                    row_data = {"风险因素": factor}
                    for i, level in enumerate(["VL", "L", "M", "H", "VH"]):
                        row_data[level] = membership[i]
                    membership_data.append(row_data)

                membership_df = pd.DataFrame(membership_data)
                membership_df.to_excel(writer, sheet_name="因素隶属度", index=False)

            # 4. 导出敏感性分析结果
            if "sensitivity_analysis" in evaluation_results:
                sensitivity = evaluation_results["sensitivity_analysis"]

                # 敏感性指标
                sens_data = []
                for factor, value in sensitivity["sensitivity_indices"].items():
                    sens_data.append({"风险因素": factor, "敏感性指标": value})

                sens_df = pd.DataFrame(sens_data)
                sens_df = sens_df.sort_values(by="敏感性指标", ascending=False)
                sens_df.to_excel(writer, sheet_name="敏感性指标", index=False)

                # 关键风险因素
                if "critical_factors" in sensitivity:
                    critical_df = pd.DataFrame({"关键风险因素": sensitivity["critical_factors"]})
                    critical_df.to_excel(writer, sheet_name="关键风险因素", index=False)

        logging.info(f"已导出评价结果到: {output_path}")
        print(f"已导出评价结果到: {output_path}")

    except Exception as e:
        logging.error(f"导出评价结果出错: {str(e)}")
        print(f"导出评价结果出错: {str(e)}")


def print_evaluation_summary(evaluation_results: Dict[str, Any]) -> None:
    """
    打印评价结果摘要

    Args:
        evaluation_results: 评价结果
    """
    print("\n========== 风险评价结果摘要 ==========")

    # 打印风险等级隶属度
    if "integrated_result" in evaluation_results:
        result = evaluation_results["integrated_result"]
        levels = ["VL", "L", "M", "H", "VH"]

        print("\n风险等级隶属度:")
        level_table = [(level, f"{value:.4f}") for level, value in zip(levels, result)]
        print(tabulate(level_table, headers=["风险等级", "隶属度"], tablefmt="grid"))

    # 打印风险指数
    if "risk_index" in evaluation_results:
        risk_index = evaluation_results["risk_index"]
        print(f"\n综合风险指数: {risk_index:.4f}")

        # 风险指数解释
        if risk_index < 0.3:
            risk_level = "低风险"
        elif risk_index < 0.5:
            risk_level = "中低风险"
        elif risk_index < 0.7:
            risk_level = "中风险"
        elif risk_index < 0.9:
            risk_level = "中高风险"
        else:
            risk_level = "高风险"

        print(f"风险等级判定: {risk_level}")

    # 打印敏感性分析结果 - 关键风险因素
    if "sensitivity_analysis" in evaluation_results:
        sensitivity = evaluation_results["sensitivity_analysis"]

        print("\n关键风险因素（敏感性排序）:")
        # 取前5个最敏感的因素
        top_factors = list(sensitivity["sensitivity_indices"].items())
        top_factors.sort(key=lambda x: abs(x[1]), reverse=True)
        top_factors = top_factors[:5]

        sens_table = [(f, f"{s:.4f}") for f, s in top_factors]
        print(tabulate(sens_table, headers=["风险因素", "敏感性指标"], tablefmt="grid"))


def main():
    """主函数"""
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="商业银行信息系统信创项目风险分析工具")
    parser.add_argument("--config", default="config.json", help="配置文件路径")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    args = parser.parse_args()

    # 设置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level=log_level)

    try:
        # 加载配置
        config = load_config(args.config)
        logging.info(f"已加载配置文件: {args.config}")

        # Extract configuration parameters by category
        # 1. Path configuration
        input_dir = config.get("paths", {}).get("input_dir", "input/")
        output_dir = config.get("paths", {}).get("output_dir", "output/")
        log_dir = config.get("paths", {}).get("log_dir", "logs/")

        # 2. File naming conventions
        ahp_excel_prefix = config.get("files", {}).get("ahp_excel_prefix", "ahp_template_")
        ahp_results_prefix = config.get("files", {}).get("ahp_results_prefix", "ahp_results_")
        fuzzy_excel = config.get("files", {}).get("fuzzy_excel", "fuzzy_risk_evaluation.xlsx")

        # 3. AHP methodological parameters
        ahp_settings = config.get("ahp_settings", {})
        ahp_model_level = ahp_settings.get("ahp_model_level")
        correction_method = ahp_settings.get("correction_method", "LLSM")
        aggregation_method = ahp_settings.get("aggregation_method", "geometric")
        weight_method = ahp_settings.get("weight_method", "eigenvector")

        # 4. Fuzzy evaluation settings
        fuzzy_settings = config.get("fuzzy_settings", {})
        use_dynamic_membership = fuzzy_settings.get("use_dynamic_membership", True)
        weight_threshold = fuzzy_settings.get("weight_threshold", 0.03)
        risk_levels = fuzzy_settings.get("risk_levels", ["VL", "L", "M", "H", "VH"])

        # 5. Analysis options
        analysis_options = config.get("analysis_options", {})
        perform_sensitivity = analysis_options.get("perform_sensitivity_analysis", True)
        sensitivity_depth = analysis_options.get("sensitivity_depth", 2)
        cross_sensitivity = analysis_options.get("cross_sensitivity_enabled", True)

        # 6. Visualization configuration
        visualization = config.get("visualization", {})
        visualization_enabled = visualization.get("enabled", True)
        output_formats = visualization.get("output_formats", ["png"])
        dpi = visualization.get("dpi", 300)

        # 初始化Excel处理器
        excel_handler = ExcelDataHandler()

        print("\n========== AHP层次分析 ==========")
        goal_excel_path = f"{input_dir}{ahp_excel_prefix}Goal.xlsx"
        ahp_expert_weights = excel_handler.read_expert_weights(goal_excel_path)
        if ahp_expert_weights:
            logging.info(f"已读取AHP专家权重: {ahp_expert_weights}")
        else:
            expert_count = ahp_settings.get("expert_count", 5)
            ahp_expert_weights = [1.0/expert_count] * expert_count
            logging.warning("读取专家权重失败，将使用均等权重")

        # 初始化AHP处理器
        ahp_processor = AHPProcessor(f'{input_dir}{ahp_excel_prefix}', ahp_expert_weights)
        ahp_processor.config = {
            "correction_method": correction_method,
            "aggregation_method": aggregation_method,
            "weight_method": weight_method,
            "output_dir": output_dir
        }

        # 处理每个AHP层级
        criteria_weights = {} # 一级风险因素局部权重
        sub_criteria_weights = {} # 二级风险因素局部权重

        for level in tqdm((ahp_model_level), desc="处理AHP层级"):
            process_ahp_level(ahp_processor, level, criteria_weights, sub_criteria_weights, f"{output_dir}{ahp_results_prefix}{level}.xlsx")

        # 计算全局权重
        global_weights = calculate_global_weights(criteria_weights, sub_criteria_weights)

        # 打印权重结果
        print("\n========== 权重分析结果 ==========")
        present_hierarchical_weights(criteria_weights, sub_criteria_weights, global_weights)

        # 执行增强型模糊综合评价
        print("\n========== 模糊综合评价 ==========")
        # Extract fuzzy evaluation configuration parameters
        fuzzy_config = {
            # Core fuzzy evaluation parameters
            "fuzzy_excel": f"{input_dir}{fuzzy_excel}",
            "use_dynamic_membership": use_dynamic_membership,
            "weight_threshold": weight_threshold,
            "risk_levels": risk_levels,

            # Sensitivity analysis configuration
            "perform_sensitivity_analysis": perform_sensitivity,
            "sensitivity_depth": sensitivity_depth,
            "cross_sensitivity_enabled": cross_sensitivity,

            # Output configuration
            "output_dir": output_dir,
            "visualization_enabled": visualization_enabled,
            "dpi": dpi
        }
        evaluation_results, significant_factors = perform_enhanced_fuzzy_evaluation(
            global_weights,
            excel_handler,
            fuzzy_config
        )

        # 打印评价结果摘要
        print_evaluation_summary(evaluation_results)

        # 可视化结果
        sorted_criteria_weights = dict(sorted(criteria_weights.items(), key=lambda item: item[1], reverse=True))
        visualize_results_(sorted_criteria_weights, significant_factors,
                           np.array(evaluation_results["integrated_result"]))

        #visualize_results(evaluation_results, config, output_dir)

        # 导出评价结果
        export_evaluation_results(evaluation_results, output_dir)

        print("\n分析完成！结果已保存到输出目录。")

    except Exception as e:
        logging.error(f"主程序执行出错: {str(e)}", exc_info=True)
        print(f"主程序执行出错: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)