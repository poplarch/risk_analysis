# visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    """结果可视化模块"""
    @staticmethod
    def plot_weights(weights_dict: Dict[str, float], title: str, output_path: str):
        """绘制权重分布图"""
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(weights_dict.keys()), y=list(weights_dict.values()))
        plt.title(title)
        plt.xticks(rotation=45)
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def plot_fuzzy_result(fuzzy_result: np.ndarray, output_path: str):
        """绘制模糊隶属度图"""
        plt.figure(figsize=(8, 5))
        plt.bar(["VL", "L", "M", "H", "VH"], fuzzy_result)
        plt.title("模糊综合评价结果")
        plt.savefig(output_path)
        plt.close()
