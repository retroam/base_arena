# Public tests for fellowship_coding_test.ipynb.
import __main__
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import nn


@dataclass
class ScoredTest:
    name: str
    points: int
    fn: Callable[[], None]


def load_solutions(*required_names):
    missing_names = [name for name in required_names if not hasattr(__main__, name)]
    if missing_names:
        missing_list = ', '.join(missing_names)
        raise AssertionError(
            f'Missing solution definitions: {missing_list}. '
            'Run the code stub cell for this question before running the tests.'
        )
    return __main__


def _assert_close(actual, expected, *, atol=1e-6):
    if not torch.allclose(actual, expected, atol=atol, rtol=0):
        raise AssertionError(f'Expected\n{expected}\nbut got\n{actual}')


def _run_scored_tests(tests: Iterable[ScoredTest], *, raise_on_failure: bool = True):
    tests = list(tests)
    earned = 0
    total = sum(test.points for test in tests)
    failures = []

    for test in tests:
        try:
            test.fn()
        except Exception as exc:  # noqa: BLE001 - helpful public test output in notebooks.
            failures.append((test, exc, traceback.format_exc()))
            print(f'✗ {test.name} (+0/{test.points})')
            print(f'  {type(exc).__name__}: {exc}')
        else:
            earned += test.points
            print(f'✓ {test.name} (+{test.points}/{test.points})')

    percent = 100 * earned / total if total else 0
    print(f'\nScore for this section: {earned}/{total} ({percent:.1f}%)')

    if failures and raise_on_failure:
        first = failures[0][2]
        raise AssertionError(
            'Some tests failed. Read the output above, fix your code, and run again.\n\n'
            'First failure traceback:\n' + first
        )
    return earned, total


# Question 1: IntegerContainerImpl — 15 points.
def _integer_adds_return_count():
    solutions = load_solutions('IntegerContainerImpl')
    container = solutions.IntegerContainerImpl()
    assert container.add(10) == 1
    assert container.add(100) == 2
    assert container.add(-5) == 3


def _integer_deletes_present_and_missing_values():
    solutions = load_solutions('IntegerContainerImpl')
    container = solutions.IntegerContainerImpl()
    for value in [10, 100, 20]:
        container.add(value)
    assert container.delete(100) is True
    assert container.delete(100) is False
    assert container.delete(999) is False
    assert container.add(5) == 3


def _integer_handles_duplicates_independently():
    solutions = load_solutions('IntegerContainerImpl')
    container = solutions.IntegerContainerImpl()
    for expected_count in range(1, 6):
        assert container.add(10) == expected_count
    for _ in range(5):
        assert container.delete(10) is True
    assert container.delete(10) is False
    assert container.add(10) == 1


def _integer_get_most_frequent_values_and_ties():
    solutions = load_solutions('IntegerContainerImpl')
    container = solutions.IntegerContainerImpl()
    assert container.get_most_frequent() is None
    container.add(5)
    container.add(5)
    container.add(3)
    assert container.get_most_frequent() == 5
    container.add(3)
    container.add(3)  # 3 now has count 3, 5 has count 2
    assert container.get_most_frequent() == 3
    # Ties are broken by the smallest value.
    tie = solutions.IntegerContainerImpl()
    tie.add(9)
    tie.add(2)
    assert tie.get_most_frequent() == 2
    tie.delete(2)
    assert tie.get_most_frequent() == 9


def _integer_get_most_frequent_is_efficient():
    solutions = load_solutions('IntegerContainerImpl')
    container = solutions.IntegerContainerImpl()
    # Interleaving adds with queries must stay fast; an O(n) rescan per query
    # (e.g. Counter(list) or max(set, key=list.count)) blows past this budget.
    num_ops = 60000
    num_distinct = 600
    start = time.perf_counter()
    for i in range(num_ops):
        container.add(i % num_distinct)
        container.get_most_frequent()
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f'get_most_frequent too slow: {elapsed:.2f}s for {num_ops} interleaved ops'


INTEGER_CONTAINER_TESTS = [
    ScoredTest('add returns the updated count', 4, _integer_adds_return_count),
    ScoredTest('delete handles present and missing values', 4, _integer_deletes_present_and_missing_values),
    ScoredTest('duplicates are stored and deleted independently', 3, _integer_handles_duplicates_independently),
    ScoredTest('get_most_frequent returns the mode and breaks ties by smallest value', 2, _integer_get_most_frequent_values_and_ties),
    ScoredTest('get_most_frequent stays efficient under many interleaved ops', 2, _integer_get_most_frequent_is_efficient),
]


# Question 2: top_k_frequent_words — 10 points.
def _top_k_basic_frequency_order():
    solutions = load_solutions('top_k_frequent_words')
    words = ['arena', 'base', 'arena', 'model', 'base', 'arena']
    assert solutions.top_k_frequent_words(words, 2) == ['arena', 'base']


def _top_k_ties_are_alphabetical():
    solutions = load_solutions('top_k_frequent_words')
    words = ['pear', 'apple', 'banana', 'banana', 'apple', 'pear']
    assert solutions.top_k_frequent_words(words, 3) == ['apple', 'banana', 'pear']


def _top_k_edge_cases():
    solutions = load_solutions('top_k_frequent_words')
    assert solutions.top_k_frequent_words(['z', 'x', 'z', 'y', 'x', 'z'], 10) == ['z', 'x', 'y']
    assert solutions.top_k_frequent_words([], 5) == []
    assert solutions.top_k_frequent_words(['a', 'b'], 0) == []


TOP_K_FREQUENT_TESTS = [
    ScoredTest('orders by descending frequency', 4, _top_k_basic_frequency_order),
    ScoredTest('breaks frequency ties alphabetically', 3, _top_k_ties_are_alphabetical),
    ScoredTest('handles k larger than unique count and empty inputs', 3, _top_k_edge_cases),
]


# Question 3: PyTorch basics — 15 points.
def _normalize_rows_values_and_zero_rows():
    solutions = load_solutions('normalize_rows')
    x = torch.tensor([[3.0, 4.0], [1.0, 2.0], [0.0, 0.0]])
    out = solutions.normalize_rows(x)
    expected = torch.tensor([[0.6, 0.8], [1 / 5**0.5, 2 / 5**0.5], [0.0, 0.0]])
    _assert_close(out, expected)


def _normalize_rows_shape_no_mutation_and_gradients():
    solutions = load_solutions('normalize_rows')
    torch.manual_seed(0)
    x = torch.randn(4, 5, requires_grad=True)
    original = x.detach().clone()
    out = solutions.normalize_rows(x)
    assert out.shape == x.shape
    _assert_close(x.detach(), original)
    _assert_close(out.detach().norm(dim=1), torch.ones(4))
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def _make_tiny_mlp_structure_and_forward():
    solutions = load_solutions('make_tiny_mlp')
    torch.manual_seed(123)
    model = solutions.make_tiny_mlp(input_dim=3, hidden_dim=4, output_dim=2)
    assert isinstance(model, nn.Module)
    layers = list(model.children()) if isinstance(model, nn.Sequential) else list(model.modules())[1:]
    assert any(isinstance(layer, nn.ReLU) for layer in layers)
    linear_layers = [layer for layer in layers if isinstance(layer, nn.Linear)]
    assert len(linear_layers) == 2
    assert linear_layers[0].in_features == 3
    assert linear_layers[0].out_features == 4
    assert linear_layers[1].in_features == 4
    assert linear_layers[1].out_features == 2
    x = torch.randn(5, 3)
    y = model(x)
    assert y.shape == (5, 2)


PYTORCH_BASICS_TESTS = [
    ScoredTest('normalize_rows returns correct values and preserves zero rows', 5, _normalize_rows_values_and_zero_rows),
    ScoredTest('normalize_rows preserves shape, input values, and gradients', 5, _normalize_rows_shape_no_mutation_and_gradients),
    ScoredTest('make_tiny_mlp returns the requested module and forward shape', 5, _make_tiny_mlp_structure_and_forward),
]


# Question 4: masked_softmax — 20 points.
def _masked_softmax_bool_basic_values():
    solutions = load_solutions('masked_softmax')
    scores = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, True]])
    out = solutions.masked_softmax(scores, mask)
    expected = torch.tensor([[0.11920292, 0.0, 0.88079708]])
    _assert_close(out, expected)


def _masked_softmax_bool_broadcast_and_dim():
    solutions = load_solutions('masked_softmax')
    scores = torch.tensor([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False],
                         [True, False, False]])
    out = solutions.masked_softmax(scores, mask, dim=0)
    expected = torch.tensor([[0.04742587, 1.0, 0.0],
                             [0.95257413, 0.0, 0.0]])
    _assert_close(out, expected)
    _assert_close(out.sum(dim=0), torch.tensor([1.0, 1.0, 0.0]))
    out.sum().backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def _masked_softmax_handles_all_masked_rows_without_nan():
    solutions = load_solutions('masked_softmax')
    scores = torch.tensor([[10.0, 0.0, -10.0], [1.0, 2.0, 3.0]])
    mask = torch.tensor([[False, False, False], [False, True, False]])
    out = solutions.masked_softmax(scores, mask)
    assert not torch.isnan(out).any()
    _assert_close(out[0], torch.zeros(3))
    _assert_close(out[1], torch.tensor([0.0, 1.0, 0.0]))


def _masked_softmax_float_mask_finite_values_are_additive():
    solutions = load_solutions('masked_softmax')
    scores = torch.tensor([[1.0, 2.0, 3.0]])
    additive = torch.tensor([[0.0, -1.0, 0.0]])
    out = solutions.masked_softmax(scores, additive)
    expected = torch.softmax(torch.tensor([[1.0, 1.0, 3.0]]), dim=-1)
    _assert_close(out, expected)
    assert torch.all(out > 0), 'finite additive mask values should not zero positions'


def _masked_softmax_float_mask_neginf_and_all_masked_slices():
    solutions = load_solutions('masked_softmax')
    scores = torch.tensor([[1.0, 2.0, 3.0],
                           [1.0, 2.0, 3.0]], requires_grad=True)
    additive = torch.tensor([[0.0, float('-inf'), 0.0],
                             [float('-inf'), float('-inf'), float('-inf')]])
    out = solutions.masked_softmax(scores, additive)
    expected = torch.tensor([[0.11920292, 0.0, 0.88079708],
                             [0.0, 0.0, 0.0]])
    assert not torch.isnan(out).any()
    _assert_close(out, expected)
    out.sum().backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


MASKED_SOFTMAX_TESTS = [
    ScoredTest('boolean mask matches expected values', 4, _masked_softmax_bool_basic_values),
    ScoredTest('boolean mask honors broadcasting, dim, and finite gradients', 4, _masked_softmax_bool_broadcast_and_dim),
    ScoredTest('all-masked rows return zeros without NaNs', 4, _masked_softmax_handles_all_masked_rows_without_nan),
    ScoredTest('float masks are additive, including finite negative values', 5,
               _masked_softmax_float_mask_finite_values_are_additive),
    ScoredTest('float -inf masks zero positions and all-masked slices without NaNs', 3,
               _masked_softmax_float_mask_neginf_and_all_masked_slices),
]


# Question 5: top_p_probs — 20 points.
# Expected outputs below are precomputed with a numerically stable reference
# implementation. Match the documented behavior rather than any single library.
def _top_p_probs_sums_and_shape():
    solutions = load_solutions('top_p_probs')
    logits = torch.tensor([4.0, 3.0, 1.0, 0.0])
    out = solutions.top_p_probs(logits, top_p=0.8, temperature=1.0)
    assert out.shape == logits.shape
    _assert_close(out.sum(), torch.tensor(1.0))
    assert torch.all(out >= 0)


def _top_p_probs_matches_reference_and_preserves_order():
    solutions = load_solutions('top_p_probs')
    logits = torch.tensor([0.0, 5.0, 1.0, 4.0, -1.0])
    out = solutions.top_p_probs(logits, top_p=0.85, temperature=1.0)
    expected = torch.tensor([0.0, 0.73105860, 0.0, 0.26894143, 0.0])
    _assert_close(out, expected)
    assert out[1] > 0
    assert out[3] > 0
    assert out[0] == 0


def _top_p_probs_min_tokens_floor_expands_tiny_nucleus():
    solutions = load_solutions('top_p_probs')
    logits = torch.tensor([4.0, 3.0, 0.0, -1.0])
    out = solutions.top_p_probs(
        logits,
        top_p=0.2,
        temperature=1.0,
        min_tokens_to_keep=2,
    )
    expected = torch.tensor([0.73105860, 0.26894142, 0.0, 0.0])
    _assert_close(out, expected)
    assert torch.count_nonzero(out).item() == 2


def _top_p_probs_top_p_can_keep_more_than_minimum():
    solutions = load_solutions('top_p_probs')
    logits = torch.tensor([2.0, 1.0, 0.0])
    out = solutions.top_p_probs(
        logits,
        top_p=0.95,
        temperature=1.0,
        min_tokens_to_keep=1,
    )
    expected = torch.tensor([0.66524096, 0.24472848, 0.09003057])
    _assert_close(out, expected)
    assert torch.count_nonzero(out).item() == 3


def _top_p_probs_validates_min_tokens_and_caps_at_vocab_size():
    solutions = load_solutions('top_p_probs')
    logits = torch.tensor([4.0, 3.0, 0.0])
    out = solutions.top_p_probs(logits, top_p=0.01, min_tokens_to_keep=10)
    expected = torch.softmax(logits, dim=-1)
    _assert_close(out, expected)

    try:
        solutions.top_p_probs(logits, min_tokens_to_keep=0)
    except ValueError:
        pass
    else:
        raise AssertionError('min_tokens_to_keep < 1 should raise ValueError')

    for kwargs in [{'temperature': 0.0}, {'temperature': -1.0}, {'top_p': 0.0}, {'top_p': 1.1}]:
        try:
            solutions.top_p_probs(logits, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Expected ValueError for {kwargs}')


TOP_P_PROBS_TESTS = [
    ScoredTest('returns a same-shape probability vector that sums to 1', 3, _top_p_probs_sums_and_shape),
    ScoredTest('matches top-p filtering and preserves token order', 4, _top_p_probs_matches_reference_and_preserves_order),
    ScoredTest('min_tokens_to_keep expands a tiny nucleus', 5,
               _top_p_probs_min_tokens_floor_expands_tiny_nucleus),
    ScoredTest('top_p can keep more tokens than the minimum floor', 4,
               _top_p_probs_top_p_can_keep_more_than_minimum),
    ScoredTest('validates min_tokens_to_keep and caps it at vocab size', 4,
               _top_p_probs_validates_min_tokens_and_caps_at_vocab_size),
]


# Question 6: apply_bpe_merge and bpe_encode — 20 points.
def _bpe_merge_basic_left_to_right():
    solutions = load_solutions('apply_bpe_merge')
    assert solutions.apply_bpe_merge(['a', 'b', 'a', 'b'], ('a', 'b')) == ['ab', 'ab']
    assert solutions.apply_bpe_merge(['l', 'o', 'w', 'e', 's', 't'], ('e', 's')) == ['l', 'o', 'w', 'es', 't']


def _bpe_merge_non_overlapping_single_pass():
    solutions = load_solutions('apply_bpe_merge')
    assert solutions.apply_bpe_merge(['a', 'a', 'a', 'a'], ('a', 'a')) == ['aa', 'aa']
    assert solutions.apply_bpe_merge(['a', 'a', 'a'], ('a', 'a')) == ['aa', 'a']
    assert solutions.apply_bpe_merge(['a', 'a', 'a', 'a', 'a'], ('a', 'a')) == ['aa', 'aa', 'a']


def _bpe_merge_edge_cases_no_mutation():
    solutions = load_solutions('apply_bpe_merge')
    assert solutions.apply_bpe_merge([], ('a', 'b')) == []
    assert solutions.apply_bpe_merge(['x', 'y', 'z'], ('a', 'b')) == ['x', 'y', 'z']
    original = ['a', 'a', 'a']
    solutions.apply_bpe_merge(original, ('a', 'a'))
    assert original == ['a', 'a', 'a'], 'input list must not be mutated'


def _bpe_encode_priority_loop_beats_single_pass_list_order():
    solutions = load_solutions('bpe_encode')
    tokens = ['a', 'b', 'c']
    merges = [('ab', 'c'), ('a', 'b')]
    assert solutions.bpe_encode(tokens, merges) == ['abc']
    assert tokens == ['a', 'b', 'c'], 'input list must not be mutated'


def _bpe_encode_cascades_merged_tokens():
    solutions = load_solutions('bpe_encode')
    assert solutions.bpe_encode(['a', 'b', 'c', 'a', 'b'], [('a', 'b'), ('ab', 'c')]) == ['abc', 'ab']
    assert solutions.bpe_encode(['a', 'a', 'a', 'a'], [('a', 'a'), ('aa', 'aa')]) == ['aaaa']


def _bpe_encode_empty_no_match_and_terminates():
    solutions = load_solutions('bpe_encode')
    assert solutions.bpe_encode(['a', 'b'], []) == ['a', 'b']
    assert solutions.bpe_encode(['x', 'y', 'z'], [('a', 'b'), ('b', 'c')]) == ['x', 'y', 'z']
    assert solutions.bpe_encode([], [('a', 'b')]) == []


BPE_MERGE_TESTS = [
    ScoredTest('apply_bpe_merge merges adjacent pairs left to right', 3, _bpe_merge_basic_left_to_right),
    ScoredTest('apply_bpe_merge is non-overlapping and single-pass', 3, _bpe_merge_non_overlapping_single_pass),
    ScoredTest('apply_bpe_merge handles edge cases without mutating input', 2, _bpe_merge_edge_cases_no_mutation),
    ScoredTest('bpe_encode uses ranked iterative BPE, not one-pass list order', 5,
               _bpe_encode_priority_loop_beats_single_pass_list_order),
    ScoredTest('bpe_encode lets merged tokens cascade into later merges', 4,
               _bpe_encode_cascades_merged_tokens),
    ScoredTest('bpe_encode handles empty/no-match cases and terminates', 3,
               _bpe_encode_empty_no_match_and_terminates),
]


ALL_TEST_GROUPS = [
    ('Question 1 — Integer container', INTEGER_CONTAINER_TESTS),
    ('Question 2 — Top-k frequent words', TOP_K_FREQUENT_TESTS),
    ('Question 3 — Basic PyTorch', PYTORCH_BASICS_TESTS),
    ('Question 4 — Masked attention softmax', MASKED_SOFTMAX_TESTS),
    ('Question 5 — Temperature and top-p probabilities', TOP_P_PROBS_TESTS),
    ('Question 6 — Byte-pair merge step', BPE_MERGE_TESTS),
]


def _score_band(score_percent):
    if score_percent >= 90:
        return 'strong readiness'
    if score_percent >= 75:
        return 'ready for the fellowship'
    if score_percent >= 60:
        return 'promising foundation; extra prep recommended'
    return 'more Python/PyTorch practice recommended before the fellowship pace'


def run_integer_container_tests():
    return _run_scored_tests(INTEGER_CONTAINER_TESTS)


def run_top_k_frequent_tests():
    return _run_scored_tests(TOP_K_FREQUENT_TESTS)


def run_pytorch_basics_tests():
    return _run_scored_tests(PYTORCH_BASICS_TESTS)


def run_masked_softmax_tests():
    return _run_scored_tests(MASKED_SOFTMAX_TESTS)


def run_top_p_probs_tests():
    return _run_scored_tests(TOP_P_PROBS_TESTS)


def run_bpe_merge_tests():
    return _run_scored_tests(BPE_MERGE_TESTS)


def run_all_tests():
    earned = 0
    total = 0
    failed = False
    print('BASE Fellows public test score\n')
    for group_name, tests in ALL_TEST_GROUPS:
        print('=' * len(group_name))
        print(group_name)
        print('=' * len(group_name))
        section_earned, section_total = _run_scored_tests(tests, raise_on_failure=False)
        earned += section_earned
        total += section_total
        if section_earned != section_total:
            failed = True
        print()

    percent = 100 * earned / total if total else 0
    print(f'Total public score: {earned}/{total} ({percent:.1f}%)')
    print(f'Score band: {_score_band(percent)}')
    if failed:
        print('Some tests failed. Partial credit is shown above; fix what you can and submit when ready.')
    return earned, total
