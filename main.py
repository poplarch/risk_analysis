# -*- coding: utf-8 -*-
import os
import json
import logging
import argparse
from typing import Dict
from tabulate import tabulate
from ahp_processor import AHPProcessor, process_and_export_ahp_level
from fuzzy_evaluator import perform_fuzzy_evaluation
from excel_handler import ExcelDataHandler, ExcelExporter
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    filename="logs/risk_analysis.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def validate_config(config: Dict) -> Dict:
    """验证配置文件"""
    if not os.path.exists(config["fuzzy_excel"]):
        raise FileNotFoundError(f模糊评价文件未找到: {config["fuzzy_excel"]}")
    return config

def print_weights(weights_dict: Dict[str, float], title: str) -> None:
    """格式化打印权重信息"""
    print(f"\n{title}:")
    print(tabulate([(k, f"{v:.4f}") for k, v in weights_dict.items()], headers=["风险因素", "权重"], tablefmt="pretty"))

def main():
    """主执行函数，协调 AHP 和模糊综合评价的运行"""
    parser = argparse.ArgumentParser(description="商业银行信息系统信创项目风险分析工具")
    parser.add_argument("--config", default="config.json", help="配置文件路径")
    args = parser.parse_args()

    try:
        # 加载配置文件
        if os.path.exists(args.config):
            with open(args.config, "r") as f:
                config = json.load(f)
        else:
            config = {
                "ahp_excel_prefix": "ahp_template",
                "fuzzy_excel_path": "risk_evaluation.xlsx",
                "output_dir": "output"
            }
            with open(args.config, "w") as f:
                json.dump(config, f)

        # 创建输出目录
        os.makedirs(config["output_dir"], exist_ok=True)

        # 定义 AHP 层次模型
        ahp_model = {
            "Level": ["Goal", "技术风险C1", "业务风险C2", "管理风险C3", "运维风险C4", "供应链风险C5", "合规风险C6"],
            "criteria_weights": {},
            "sub_criteria_weights": {},
            "global_weights": {}
        }
        criteria_weights = ahp_model["criteria_weights"]
        sub_criteria_weights = ahp_model["sub_criteria_weights"]
        global_weights = ahp_model["global_weights"]
        
        # 读取全局专家权重
        excel_handler = ExcelDataHandler(None)
        global_expert_weights = excel_handler.read_expert_weights(f"{config['ahp_excel_prefix']}_Goal.xlsx")

        # 处理每个层次的 AHP 数据
	for level in tqdm(ahp_model["Level"], desc="Processing AHP Levels"):
            try:
	        process_and_export_ahp_level(level, f"{config['ahp_excel_prefix']}_{level.replace(' ', '')}.xlsx", 
	                                     criteria_weights, sub_criteria_weights, global_expert_weights)
	    except Exception as e:
	        logging.error(f"处理层级 {level} 出错，跳过: {str(e)}")
	        continue

        # 计算全局权重
        for criteria, local_weights in sub_criteria_weights.items():
            for sub_criteria, local_weight in local_weights.items():
                global_weights[sub_criteria] = local_weight * criteria_weights.get(criteria, 1.0)

        # 打印权重结果
        print_weights(criteria_weights, "一级风险因素局部权重")
        for main_criterion, sub_criteria in sub_criteria_weights.items():
            sorted_local_weights = sorted(sub_criteria.items(), key=lambda x: x[1], reverse=True)
            print_weights(dict(sorted_local_weights), f"{main_criterion} - 二级风险因素局部权重")
        print_weights(global_weights, "所有二级风险因素的全局权重")

        # 筛选全局权重大于 0.03 的风险因素进行模糊评价
        sorted_global_weights = sorted(global_weights.items(), key=lambda x: x[1], reverse=True)
        fuzzy_global_weights = dict(sorted_global_weights)
        for criterion, weight in sorted_global_weights:
            if weight < 0.03:
                del fuzzy_global_weights[criterion]

        # 执行模糊综合评价
        fuzzy_result = perform_fuzzy_evaluation(fuzzy_global_weights, config["fuzzy_excel_path"], excel_handler)
        if fuzzy_result is not None:
            excel_exporter = ExcelExporter()
            fuzzy_results = {"fuzzy_result": fuzzy_result}
            excel_exporter.export_fuzzy_results(fuzzy_results, os.path.join(config["output_dir"], "fce_risk_analysis_results.xlsx"))
            print("\n模糊综合评价结果:")
            print(tabulate([(level, f"{value:.4f}") for level, value in zip(["VL", "L", "M", "H", "VH"], fuzzy_result)],
                           headers=["风险等级", "隶属度"], tablefmt="pretty"))

    except Exception as e:
        logging.error(f"主程序执行出错: {str(e)}")
        print(f"主程序执行出错: {str(e)}")

if __name__ == "__main__":
    main()
