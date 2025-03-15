# 项目风险分析工具 Risk Analysis Tool

## 简介
用于商业银行信息系统信创项目的风险分析工具，支持 AHP 和模糊综合评价，可开展风险敏感性分析，包括单因素和双因素敏感性分析。

可支持专家判断矩阵的自动修正（对数最小二乘法（LLSM）、迭代修正法、直接投影法，默认采用迭代修正法，可以最大化保留专家意见）。

可生成一级和二级风险因素局部权重条形图、一级风险因素局部权重和二级风险因素全局权重饼状图、风险隶属度函数模型图、模糊综合评价隶属度柱状图、风险因素敏感性雷达图、风险因素权重变化敏感性曲线、风险影响热力图、关键风险因素敏感性分析直方图。

## 安装依赖库
pip install -r requirements.txt

## 使用说明
1. 生成AHP专家判断矩阵数据收集模板
ahp-excel-template-generator.py

2. 生成FCE专家风险评测结果数据收集模板
fce-excel-template-generator.py

3. 执行数据分析
python main.py --config config.json
