# -*- coding: utf-8 -*-
import os
import time
import pandas as pd
import numpy as np
import logging
from openpyxl.styles import Alignment
from typing import List, Dict, Tuple, Optional

class ExcelExporter:
    """Excel 文件导出器，用于格式化输出结果"""
    def adjust_all_sheets(self, writer: pd.ExcelWriter) -> None:
        """调整所有工作表的列宽和对齐方式"""
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column_cells in worksheet.columns:
                max_length = max(len(str(cell.value)) if cell.value else 0 for cell in column_cells)
                column_letter = column_cells[0].column_letter
                worksheet.column_dimensions[column_letter].width = max_length + 8
                for cell in column_cells:
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    @staticmethod
    def matrix_compare_and_highlight(original_matrix: np.ndarray, corrected_matrix: np.ndarray, writer: pd.ExcelWriter, sheet_name: str, criteria: List[str]) -> None:
        """比较并高亮显示原始和修正矩阵的差异"""
        from openpyxl.styles import PatternFill
        from openpyxl.utils import get_column_letter
        n = len(original_matrix)
        df1 = pd.DataFrame(np.round(original_matrix, 4), columns=criteria, index=criteria)
        df2 = pd.DataFrame(np.round(corrected_matrix, 4), columns=criteria, index=criteria)
        df1.to_excel(writer, sheet_name=sheet_name, startrow=1, startcol=0)
        offset = n + 3
        df2.to_excel(writer, sheet_name=sheet_name, startrow=1, startcol=offset)
        ws = writer.sheets[sheet_name]
        ws["A1"] = "原始矩阵"
        ws[f"{get_column_letter(offset + 1)}1"] = "修正矩阵"
        grey_fill = PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid")
        for i in range(n):
            for j in range(n):
                if abs(original_matrix[i, j] - corrected_matrix[i, j]) > 1e-4:
                    cell1 = f"{get_column_letter(j + 2)}{i + 3}"
                    cell2 = f"{get_column_letter(offset + j + 2)}{i + 3}"
                    ws[cell1].fill = grey_fill
                    ws[cell2].fill = grey_fill

    def export_ahp_results(self, results: Dict, output_path: str, include_matrices: bool = True) -> None:
        """导出 AHP 分析结果"""
        output_path = f"{os.path.splitext(output_path)[0]}_{int(time.time())}.xlsx"
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            summary_data = {
                "分析指标": ["专家数量", "评价准则数量", "群组矩阵一致性比率(CR)", "修正判断矩阵数量"],
                "结果值": [len(results["corrected_matrices"]), len(results["final_weights"]),
                          round(results["aggregated_consistency"].consistency_ratio, 4),
                          sum(1 for corr in results["correction_results"] if corr.adjusted)]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="汇总信息", index=False)
            weights_data = [{"准则名": criterion, "权重值": round(weight, 4)} for criterion, weight in results["final_weights"].items()]
            pd.DataFrame(weights_data).to_excel(writer, sheet_name="最终权重", index=False)
            consistency_data = [
                {"专家编号": f"专家{corr.expert_id}", "原始CR值": round(corr.original_cr, 4),
                 "修正后CR值": round(corr.final_cr, 4), "是否达到一致性": "是" if corr.success else "否",
                 "是否动态修正": "是" if corr.adjusted else "否"}
                for corr in results["correction_results"]
            ]
            pd.DataFrame(consistency_data).to_excel(writer, sheet_name="专家一致性分析结果", index=False)
            if include_matrices:
                criteria = list(results["final_weights"].keys())
                aggregated_df = pd.DataFrame(np.round(results["aggregated_matrix"], 4), index=criteria, columns=criteria)
                aggregated_df.to_excel(writer, sheet_name="群组判断矩阵")
                for i, (original, corrected) in enumerate(zip(results["original_matrices"], results["corrected_matrices"])):
                    sheet_name = f"Expert{i+1}_Comparison"
                    self.matrix_compare_and_highlight(original, corrected, writer, sheet_name, criteria)
            self.adjust_all_sheets(writer)

    def export_fuzzy_results(self, fuzzy_results: Dict, output_path: str) -> None:
        """导出模糊综合评价结果"""
        output_path = f"{os.path.splitext(output_path)[0]}_{int(time.time())}.xlsx"
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            fuzzy_df = pd.DataFrame(fuzzy_results["fuzzy_result"].reshape(1, -1).round(4),
                                    index=["风险等级"], columns=["VL", "L", "M", "H", "VH"])
            fuzzy_df.to_excel(writer, sheet_name="模糊评价结果")
            self.adjust_all_sheets(writer)

class ExcelDataHandler:
    """Excel 文件数据处理器，负责读取 AHP 和模糊评价数据"""
    def __init__(self, matrix_validator=None):
        self.matrix_validator = matrix_validator

    def read_expert_matrices(self, excel_path: str, sheet_prefix: str = "Expert") -> Tuple[List[np.ndarray], List[str]]:
        """读取专家判断矩阵数据"""
        if not os.path.exists(excel_path):
            logging.error(f"文件未找到: {excel_path}")
            raise FileNotFoundError(f"Excel 文件未找到: {excel_path}")
        xl = pd.ExcelFile(excel_path)
        expert_sheets = [s for s in xl.sheet_names if s.startswith(sheet_prefix)]
        if not expert_sheets:
            logging.error(f"在 {excel_path} 中未找到以 '{sheet_prefix}' 开头的工作表")
            raise ValueError(f"未找到以 '{sheet_prefix}' 开头的工作表")
        matrices = []
        criteria_names = None
        for sheet in expert_sheets:
            df = pd.read_excel(excel_path, sheet_name=sheet, index_col=0)
            if criteria_names is None:
                criteria_names = df.index.tolist()
            elif df.index.tolist() != criteria_names:
                logging.error(f"{sheet} 中的准则名称与其他工作表不匹配")
                raise ValueError(f"{sheet} 中的准则名称与其他工作表不匹配")
            partial_matrix = np.zeros(df.shape)
            for i in range(len(df)):
                for j in range(i, len(df)):
                    value = df.iloc[i, j]
                    if isinstance(value, str) and '/' in value:
                        from fractions import Fraction
                        partial_matrix[i, j] = float(Fraction(value))
                    else:
                        partial_matrix[i, j] = pd.to_numeric(value, errors="coerce")
                        if np.isnan(partial_matrix[i, j]):
                            logging.error(f"在 {sheet} 的单元格 ({i+1}, {j+1}) 中发现无效值: {value}")
                            raise ValueError(f"在 {sheet} 中发现非数字内容: '{value}'")
            if self.matrix_validator:
                self.matrix_validator.validate_upper_triangular(partial_matrix, sheet)
            complete_matrix = self._complete_matrix(partial_matrix)
            matrices.append(complete_matrix)
        return matrices, criteria_names

    def read_expert_scores(self, excel_path: str) -> Tuple[pd.DataFrame, Optional[List[float]]]:
        """读取专家评分数据"""
        if not os.path.exists(excel_path):
            logging.error(f"文件未找到: {excel_path}")
            raise FileNotFoundError(f"Excel 文件未找到: {excel_path}")
        try:
            scores_df = pd.read_excel(excel_path, sheet_name="专家评分", skiprows=1, usecols=[1, 2, 3, 4, 5], index_col=0)
            if not scores_df.apply(lambda x: x.between(1, 10)).all().all():
                logging.error("评分超出 1-10 范围")
                raise ValueError("评分必须在 1-10 之间")
        except Exception as e:
            logging.error(f"读取评分数据错误: {str(e)}")
            raise
        try:
            weights_df = pd.read_excel(excel_path, sheet_name="专家权重", usecols=["权重值"])
            expert_weights = weights_df["权重值"].values.tolist()
            if not np.isclose(sum(expert_weights), 1.0, rtol=1e-5):
                logging.warning("专家权重和不为 1，自动归一化")
                expert_weights = [w / sum(expert_weights) for w in expert_weights]
        except Exception as e:
            logging.warning(f"无法读取专家权重: {str(e)}，使用默认值")
            expert_weights = None
        return scores_df, expert_weights

    @staticmethod
    def _complete_matrix(partial_matrix: np.ndarray) -> np.ndarray:
        """补全下三角矩阵"""
        n = len(partial_matrix)
        complete = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                complete[i, j] = partial_matrix[i, j]
                complete[j, i] = 1.0 / partial_matrix[i, j]
        return complete

    @staticmethod
    def read_expert_weights(excel_path: str, sheet_name: str = "expertWeights") -> Optional[List[float]]:
        """读取专家权重"""
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            weights = df["Weight"].values.tolist()
            if not np.isclose(sum(weights), 1.0, rtol=1e-5):
                logging.warning("专家权重和不为 1，自动归一化")
                weights = [w / sum(weights) for w in weights]
            return weights
        except Exception as e:
            logging.warning(f"无法读取专家权重: {str(e)}")
            return None
