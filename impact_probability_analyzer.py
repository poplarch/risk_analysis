class RiskImpactProbabilityAnalyzer:
    """风险概率-影响矩阵分析器"""

    def __init__(self, probability_levels: List[str] = None, impact_levels: List[str] = None):
        """初始化概率-影响分析器"""
        self.probability_levels = probability_levels or ["Very Low", "Low", "Medium", "High", "Very High"]
        self.impact_levels = impact_levels or ["Negligible", "Minor", "Moderate", "Major", "Critical"]
        self.risk_matrix = self._initialize_matrix()

    def _initialize_matrix(self) -> np.ndarray:
        """初始化风险矩阵"""
        n_prob = len(self.probability_levels)
        n_impact = len(self.impact_levels)

        # 创建风险矩阵 (风险等级: 1=低, 2=中低, 3=中, 4=中高, 5=高)
        matrix = np.zeros((n_prob, n_impact))

        # 根据概率和影响计算风险等级
        for i in range(n_prob):
            for j in range(n_impact):
                # 概率和影响越高，风险等级越高
                matrix[i, j] = (i + 1) * (j + 1) / (n_prob * n_impact) * 5

        return matrix

    def categorize_risks(self, risks_data: pd.DataFrame) -> pd.DataFrame:
        """
        根据概率和影响对风险进行分类
        
        Args:
            risks_data (pd.DataFrame): 包含风险ID、概率等级和影响等级的数据框
            
        Returns:
            pd.DataFrame: 包含风险等级的数据框
        """
        results = risks_data.copy()
        risk_levels = []

        for _, row in risks_data.iterrows():
            prob_idx = self.probability_levels.index(row['概率等级'])
            impact_idx = self.impact_levels.index(row['影响等级'])
            risk_score = self.risk_matrix[prob_idx, impact_idx]

            # 确定风险等级
            if risk_score < 1:
                risk_level = "极低"
            elif risk_score < 2:
                risk_level = "低"
            elif risk_score < 3:
                risk_level = "中"
            elif risk_score < 4:
                risk_level = "高"
            else:
                risk_level = "极高"

            risk_levels.append({"风险等级": risk_level, "风险分数": risk_score})

        results = pd.concat([results, pd.DataFrame(risk_levels)], axis=1)
        return results

    def visualize_risk_matrix(self, risks_data: pd.DataFrame, output_path: str):
        """
        可视化风险矩阵
        
        Args:
            risks_data (pd.DataFrame): 风险数据
            output_path (str): 输出路径
        """
        # 创建热力图底图
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.risk_matrix, annot=True, cmap="RdYlGn_r",
                    xticklabels=self.impact_levels,
                    yticklabels=self.probability_levels)

        # 添加风险点标注
        for _, risk in risks_data.iterrows():
            prob_idx = self.probability_levels.index(risk['概率等级'])
            impact_idx = self.impact_levels.index(risk['影响等级'])
            plt.plot(impact_idx + 0.5, prob_idx + 0.5, 'ko', markersize=8)
            plt.annotate(risk['风险ID'], (impact_idx + 0.7, prob_idx + 0.5),
                         color='black', fontweight='bold')

        plt.title("风险概率-影响矩阵")
        plt.xlabel("影响程度")
        plt.ylabel("发生概率")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
