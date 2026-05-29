# Public tests for arena_coding_test.ipynb.
import __main__
import unittest

import torch
from torch import nn


def load_solutions(*required_names):
    missing_names = [name for name in required_names if not hasattr(__main__, name)]
    if missing_names:
        missing_list = ', '.join(missing_names)
        raise AssertionError(
            f'Missing solution definitions: {missing_list}. '
            'Run the code stub cell for this question before running the tests.'
        )
    return __main__


class IntegerContainerTests(unittest.TestCase):
    def setUp(self):
        solutions = load_solutions('IntegerContainerImpl')
        self.container = solutions.IntegerContainerImpl()

    def test_add_two_numbers(self):
        self.assertEqual(self.container.add(10), 1)
        self.assertEqual(self.container.add(100), 2)

    def test_add_many_numbers(self):
        for expected_count, value in enumerate([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], start=1):
            self.assertEqual(self.container.add(value), expected_count)

    def test_delete_number(self):
        self.assertEqual(self.container.add(10), 1)
        self.assertEqual(self.container.add(100), 2)
        self.assertTrue(self.container.delete(10))

    def test_delete_nonexisting_number(self):
        self.assertEqual(self.container.add(10), 1)
        self.assertEqual(self.container.add(100), 2)
        self.assertFalse(self.container.delete(20))
        self.assertTrue(self.container.delete(10))
        self.assertFalse(self.container.delete(10))

    def test_add_and_delete_duplicate_numbers(self):
        for expected_count in range(1, 6):
            self.assertEqual(self.container.add(10), expected_count)
        for _ in range(5):
            self.assertTrue(self.container.delete(10))
        self.assertFalse(self.container.delete(10))

    def test_delete_in_random_order(self):
        for expected_count, value in enumerate([10, 20, 30, 40, 40], start=1):
            self.assertEqual(self.container.add(value), expected_count)
        self.assertTrue(self.container.delete(30))
        self.assertFalse(self.container.delete(30))
        self.assertTrue(self.container.delete(10))
        self.assertFalse(self.container.delete(10))
        self.assertTrue(self.container.delete(40))
        self.assertTrue(self.container.delete(40))
        self.assertFalse(self.container.delete(40))
        self.assertTrue(self.container.delete(20))
        self.assertFalse(self.container.delete(20))


class TopKFrequentTests(unittest.TestCase):
    def setUp(self):
        self.solutions = load_solutions('top_k_frequent_words')

    def test_basic_case(self):
        words = ['arena', 'base', 'arena', 'model', 'base', 'arena']
        self.assertEqual(self.solutions.top_k_frequent_words(words, 2), ['arena', 'base'])

    def test_ties_are_alphabetical(self):
        words = ['pear', 'apple', 'banana', 'banana', 'apple', 'pear']
        self.assertEqual(self.solutions.top_k_frequent_words(words, 3), ['apple', 'banana', 'pear'])

    def test_k_larger_than_unique_count(self):
        words = ['z', 'x', 'z', 'y', 'x', 'z']
        self.assertEqual(self.solutions.top_k_frequent_words(words, 10), ['z', 'x', 'y'])

    def test_empty_input(self):
        self.assertEqual(self.solutions.top_k_frequent_words([], 5), [])
        self.assertEqual(self.solutions.top_k_frequent_words(['a', 'b'], 0), [])


class PytorchTests(unittest.TestCase):
    def setUp(self):
        self.solutions = load_solutions('normalize_rows', 'make_tiny_mlp')

    def test_normalize_rows_values(self):
        x = torch.tensor([[3.0, 4.0], [1.0, 2.0], [0.0, 0.0]])
        out = self.solutions.normalize_rows(x)
        expected = torch.tensor([[0.6, 0.8], [1 / 5**0.5, 2 / 5**0.5], [0.0, 0.0]])
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_normalize_rows_shape_and_no_mutation(self):
        torch.manual_seed(0)
        x = torch.randn(4, 5)
        original = x.clone()
        out = self.solutions.normalize_rows(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.allclose(x, original))
        self.assertTrue(torch.allclose(out.norm(dim=1), torch.ones(4), atol=1e-6))

    def test_normalize_rows_preserves_gradients(self):
        x = torch.tensor([[3.0, 4.0]], requires_grad=True)
        out = self.solutions.normalize_rows(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    def test_make_tiny_mlp_structure_and_forward(self):
        torch.manual_seed(123)
        model = self.solutions.make_tiny_mlp(input_dim=3, hidden_dim=4, output_dim=2)
        self.assertIsInstance(model, nn.Module)
        layers = list(model.children()) if isinstance(model, nn.Sequential) else list(model.modules())[1:]
        self.assertTrue(any(isinstance(layer, nn.ReLU) for layer in layers))
        linear_layers = [layer for layer in layers if isinstance(layer, nn.Linear)]
        self.assertEqual(len(linear_layers), 2)
        self.assertEqual(linear_layers[0].in_features, 3)
        self.assertEqual(linear_layers[0].out_features, 4)
        self.assertEqual(linear_layers[1].in_features, 4)
        self.assertEqual(linear_layers[1].out_features, 2)
        x = torch.randn(5, 3)
        y = model(x)
        self.assertEqual(y.shape, (5, 2))


def run_test_class(test_class):
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_class)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError('Some tests failed. Read the output above, fix your code, and run again.')


def run_integer_container_tests():
    run_test_class(IntegerContainerTests)


def run_top_k_frequent_tests():
    run_test_class(TopKFrequentTests)


def run_pytorch_tests():
    run_test_class(PytorchTests)


def run_all_tests():
    run_integer_container_tests()
    run_top_k_frequent_tests()
    run_pytorch_tests()
