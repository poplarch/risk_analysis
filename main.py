# -*- coding: utf-8 -*-
# TODO: 层次分析法支持4层模型，包括目标层、准则层、指标层、方案层（可选）
# TODO: 增加GUI支持、多层次模糊综合评价法
"""
风险分析工具主程序
支持AHP和增强型FCE方法的集成风险分析系统
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from tabulate import tabulate
from tqdm import tqdm

# 导入自定义模块
from ahp_processor import AHPProcessor, ConsistencyChecker, MatrixCorrector
from excel_handler import ExcelDataHandler, ExcelExporter
from enhanced_fuzzy_evaluator import EnhancedFuzzyEvaluator
from visualizer import Visualizer


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


def validate_config(config: Dict) -> Dict:
    """
    验证配置文件

    Args:
        config: 配置字典

    Returns:
        验证后的配置字典

    Raises:
        FileNotFoundError: 如果必要文件不存在
        ValueError: 如果配置项无效
    """
    required_files = [
        ("fuzzy_excel", "模糊评价数据文件"),
        ("input_dir", "输入目录")
    ]

    # 检查文件存在性
    for file_key, description in required_files:
        if file_key in config:
            path = config[file_key]
            if file_key.endswith("_dir"):
                if not os.path.isdir(path):
                    os.makedirs(path, exist_ok=True)
                    logging.warning(f"创建{description}目录: {path}")
            elif not os.path.exists(f'{config["input_dir"]}{path}'):
                raise FileNotFoundError(f"{description}{config['input_dir']}{path}'")

    # 设置默认值
    defaults = {
        "ahp_excel_inprefix": "ahp_template",
        "ahp_excel_outprefix": "ahp_results",
        "fuzzy_excel": "fuzzy_risk_evaluation.xlsx",
        "input_dir": "input",
        "output_dir": "output",
        "use_dynamic_membership": True,
        "perform_sensitivity_analysis": True,
        "visualization_enabled": True
    }

    # 应用默认值
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
            logging.info(f"应用默认配置: {key} = {value}")

    return config


def process_ahp_level(
        ahp_processor: AHPProcessor,
        level: str,
        criteria_weights: Dict[str, float],
        sub_criteria_weights: Dict[str, Dict[str, float]]
) -> None:
    """
    处理单个AHP层级并更新权重字典

    Args:
        ahp_processor: AHP处理器
        level: 层级名称
        criteria_weights: 一级准则权重字典
        sub_criteria_weights: 二级准则权重字典
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
        output_path = os.path.join(
            ahp_processor.config["output_dir"],
            f"{ahp_processor.config['ahp_excel_outprefix']}_{level}.xlsx"
        )
        ExcelExporter().export_ahp_results(results, output_path)
        logging.info(f"已导出AHP结果到: {output_path}")

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
        fuzzy_excel_path: str,
        config: Dict
) -> Dict[str, Any]:
    """
    执行增强型模糊综合评价

    Args:
        global_weights: 全局权重字典
        excel_handler: Excel数据处理器
        fuzzy_excel_path: 模糊评价Excel文件路径
        config: 配置信息

    Returns:
        评价结果字典
    """
    try:
        # 初始化增强型模糊评价器
        fuzzy_evaluator = EnhancedFuzzyEvaluator(
            dynamic_enabled=config.get("use_dynamic_membership", True)
        )

        # 读取专家评分数据
        scores_df, expert_weights = excel_handler.read_expert_scores(fuzzy_excel_path)

        # 过滤权重大于阈值的风险因素
        weight_threshold = config.get("weight_threshold", 0.03)
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
            use_dynamic=config.get("use_dynamic_membership", True)
        )

        # 如果需要进行敏感性分析
        if config.get("perform_sensitivity_analysis", True):
            logging.info("执行敏感性分析...")

            # 单因素敏感性分析
            sensitivity_results = fuzzy_evaluator.perform_sensitivity_analysis(
                factor_weights=significant_factors,
                expert_scores=expert_scores,
                use_dynamic=config.get("use_dynamic_membership", True)
            )

            # 识别最敏感的两个因素进行交叉敏感性分析
            top_factors = sensitivity_results["ranked_factors"][:2]
            if len(top_factors) == 2:
                cross_results = fuzzy_evaluator.cross_sensitivity_analysis(
                    factor_weights=significant_factors,
                    expert_scores=expert_scores,
                    factors=top_factors,
                    use_dynamic=config.get("use_dynamic_membership", True)
                )
            else:
                cross_results = None
                logging.warning("敏感因素不足，无法进行交叉敏感性分析")

            # 合并敏感性分析结果
            evaluation_results["sensitivity_analysis"] = sensitivity_results
            if cross_results:
                evaluation_results["cross_sensitivity"] = cross_results

        return evaluation_results

    except Exception as e:
        logging.error(f"模糊综合评价出错: {str(e)}")
        raise


def visualize_results(
        evaluation_results: Dict[str, Any],
        config: Dict,
        output_dir: str
) -> None:
    """
    可视化分析结果

    Args:
        evaluation_results: 评价结果
        config: 配置信息
        output_dir: 输出目录
    """
    if not config.get("visualization_enabled", True):
        logging.info("可视化功能已禁用")
        return

    try:
        # 创建可视化目录
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # 初始化模糊评价器
        fuzzy_evaluator = EnhancedFuzzyEvaluator(
            dynamic_enabled=config.get("use_dynamic_membership", True)
        )

        # 1. 可视化隶属度函数
        if "factor_membership" in evaluation_results:
            # 获取第一个因素的评分样本用于演示
            first_factor = list(evaluation_results["factor_membership"].keys())[0]
            sample_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # 示例评分点

            fuzzy_evaluator.visualize_membership_functions(
                use_dynamic=config.get("use_dynamic_membership", True),
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

            plt.title("风险等级隶属度分布")
            plt.ylabel("隶属度")
            plt.ylim(0, max(result) * 1.2)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(viz_dir, "fuzzy_results.png"), dpi=300)
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
        logging.error(f"生成可视化图表出错: {str(e)}")
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
    parser = argparse.ArgumentParser(description="商业银行信息系统信创项目风险分析工具 - 优化版")
    parser.add_argument("--config", default="config.json", help="配置文件路径")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    args = parser.parse_args()

    # 设置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level=log_level)

    try:
        # 加载配置
        if os.path.exists(args.config):
            with open(args.config, "r", encoding="utf-8") as f:
                config = json.load(f)
            logging.info(f"已加载配置文件: {args.config}")
        else:
            # 默认配置
            config = {
                "ahp_excel_inprefix": "ahp_template_",
                "ahp_excel_outprefix": "ahp_results_",
                "fuzzy_excel": "fuzzy_risk_evaluation.xlsx",
                "input_dir": "input",
                "output_dir": "output",
                "use_dynamic_membership": True,
                "perform_sensitivity_analysis": True,
                "visualization_enabled": True,
                "weight_threshold": 0.03
            }

            # 保存默认配置
            os.makedirs(os.path.dirname(args.config) or ".", exist_ok=True)
            with open(args.config, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            logging.info(f"已创建默认配置文件: {args.config}")

        # 验证配置
        config = validate_config(config)

        # 确保输出目录存在
        os.makedirs(config["output_dir"], exist_ok=True)

        # 定义AHP层次模型结构
        ahp_model = {
            "Level": ["Goal", "技术风险C1", "业务风险C2", "管理风险C3", "运维风险C4", "供应链风险C5", "合规风险C6"],
            "criteria_weights": {},  # 一级准则权重
            "sub_criteria_weights": {},  # 二级准则权重
        }

        # 初始化Excel处理器
        excel_handler = ExcelDataHandler()

        # 读取专家权重
        global_expert_weights = None
        try:
            goal_excel_path = f"{config['input_dir']}{config['ahp_excel_inprefix']}_Goal.xlsx"
            global_expert_weights = excel_handler.read_expert_weights(goal_excel_path)
            if global_expert_weights:
                logging.info(f"已读取全局专家权重: {global_expert_weights}")
        except Exception as e:
            logging.warning(f"读取专家权重失败，将使用均等权重: {str(e)}")

        # 初始化AHP处理器
        ahp_processor = AHPProcessor(f'{config["input_dir"]}{config["ahp_excel_inprefix"]}', global_expert_weights)
        ahp_processor.config = config  # 附加配置信息

        # 处理每个AHP层级
        criteria_weights = ahp_model["criteria_weights"]
        sub_criteria_weights = ahp_model["sub_criteria_weights"]

        print("\n========== AHP层次分析 ==========")
        for level in tqdm(ahp_model["Level"], desc="处理AHP层级"):
            process_ahp_level(ahp_processor, level, criteria_weights, sub_criteria_weights)

        # 计算全局权重
        global_weights = calculate_global_weights(criteria_weights, sub_criteria_weights)

        # 打印权重结果
        print("\n========== 权重分析结果 ==========")
        print("\n一级准则权重:")
        weights_table = [(k, f"{v:.4f}") for k, v in criteria_weights.items()]
        print(tabulate(weights_table, headers=["准则", "权重"], tablefmt="grid"))

        print("\n二级准则局部权重:")
        for main_criterion, weights in sub_criteria_weights.items():
            print(f"\n{main_criterion}:")
            local_table = [(k, f"{v:.4f}") for k, v in weights.items()]
            print(tabulate(local_table, headers=["准则", "局部权重"], tablefmt="grid"))

        print("\n所有准则全局权重:")
        global_table = [(k, f"{v:.4f}") for k, v in sorted(global_weights.items(),
                                                           key=lambda x: x[1], reverse=True)]
        print(tabulate(global_table, headers=["准则", "全局权重"], tablefmt="grid"))

        # 执行增强型模糊综合评价
        print("\n========== 模糊综合评价 ==========")
        evaluation_results = perform_enhanced_fuzzy_evaluation(
            global_weights,
            excel_handler,
            config["fuzzy_excel"],
            config
        )

        # 打印评价结果摘要
        print_evaluation_summary(evaluation_results)

        # 可视化结果
        visualize_results(evaluation_results, config, config["output_dir"])

        # 导出评价结果
        export_evaluation_results(evaluation_results, config["output_dir"])

        print("\n分析完成！结果已保存到输出目录。")

    except Exception as e:
        logging.error(f"主程序执行出错: {str(e)}", exc_info=True)
        print(f"主程序执行出错: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)