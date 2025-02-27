# -*- coding: utf-8 -*-
import unittest
import numpy as np
from fuzzy_evaluator import FuzzyEvaluator

class TestFuzzy(unittest.TestCase):
    """模糊评价模块的单元测试"""
    def test_membership_degree(self):
        """测试隶属度计算"""
        evaluator = FuzzyEvaluator()
        scores = np.array([0.5, 0.6])
        result = evaluator.calculate_membership_degree(scores)
        self.assertAlmostEqual(sum(result), 1.0, places=6)

    def test_fuzzy_multiply(self):
        """测试模糊加权运算"""
        evaluator = FuzzyEvaluator()
        weights = np.array([0.5, 0.5])
        membership_matrix = np.array([[0.1, 0.2, 0.3, 0.3, 0.1], [0.0, 0.1, 0.4, 0.4, 0.1]])
        result = evaluator.fuzzy_multiply(weights, membership_matrix)
        self.assertAlmostEqual(sum(result), 1.0, places=6)

if __name__ == "__main__":
    unittest.main()
