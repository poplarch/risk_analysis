# -*- coding: utf-8 -*-
import unittest
import numpy as np
from ahp_processor import ConsistencyChecker, WeightCalculator

class TestAHP(unittest.TestCase):
    """AHP 模块的单元测试"""
    def test_consistency_checker(self):
        """测试一致性检查器"""
        checker = ConsistencyChecker()
        matrix = np.array([[1, 2], [0.5, 1]])
        result = checker.check_consistency(matrix)
        self.assertTrue(result.is_consistent)
        self.assertAlmostEqual(result.consistency_ratio, 0.0, places=6)

    def test_weight_calculator(self):
        """测试权重计算器"""
        calculator = WeightCalculator()
        matrix = np.array([[1, 2], [0.5, 1]])
        weights = calculator.calculate_weights(matrix)
        self.assertAlmostEqual(sum(weights), 1.0, places=6)
        self.assertAlmostEqual(weights[0], 0.6667, places=4)

if __name__ == "__main__":
    unittest.main()
