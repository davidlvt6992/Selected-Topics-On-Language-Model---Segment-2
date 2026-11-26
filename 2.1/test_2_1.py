#!/usr/bin/env python3
"""Tests for lab 2-1. Usage: python test_2_1.py [todo_number ...]"""

import importlib.util
import math
import sys
from pathlib import Path

LAB_PATH = Path(__file__).resolve().parent / "lab-2-1-scaling-laws.py"

WARMUP_COMPUTE_BUDGET = 5e21
MIN_SWEEP_STEPS = 6
MIN_SWEEP_SPAN = 100

DUMMY_FIT = {
    "irreducible_loss": 1.5,
    "model_coefficient": 300.0,
    "data_coefficient": 350.0,
    "model_exponent": 0.30,
    "data_exponent": 0.25,
}

NOISE_FREE_RUNS = [
    {
        "model_params": 50000000,
        "training_tokens": 1000000000000,
        "validation_loss": 3.320382,
    },
    {
        "model_params": 150000000,
        "training_tokens": 500000000000,
        "validation_loss": 2.973755,
    },
    {
        "model_params": 400000000,
        "training_tokens": 200000000000,
        "validation_loss": 2.81133,
    },
    {
        "model_params": 800000000,
        "training_tokens": 100000000000,
        "validation_loss": 2.762419,
    },
    {
        "model_params": 1500000000,
        "training_tokens": 60000000000,
        "validation_loss": 2.737202,
    },
    {
        "model_params": 3000000000,
        "training_tokens": 30000000000,
        "validation_loss": 2.771496,
    },
    {
        "model_params": 6000000000,
        "training_tokens": 15000000000,
        "validation_loss": 2.849788,
    },
    {
        "model_params": 10000000000,
        "training_tokens": 8000000000,
        "validation_loss": 2.970296,
    },
    {
        "model_params": 15000000000,
        "training_tokens": 4000000000,
        "validation_loss": 3.157364,
    },
    {
        "model_params": 20000000000,
        "training_tokens": 2000000000,
        "validation_loss": 3.398724,
    },
]


def load_lab():
    spec = importlib.util.spec_from_file_location("lab", LAB_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lab"] = module
    spec.loader.exec_module(module)
    return module


def check(name, got_fn, expected, compare=None):
    """Run got_fn(), compare against expected, print a PASS/FAIL line."""
    try:
        got = got_fn()
    except NotImplementedError:
        print(f"  [FAIL] {name}\n         not implemented yet")
        return False
    except Exception as error:
        print(f"  [FAIL] {name}\n         raised {type(error).__name__}: {error}")
        return False
    ok = compare(got, expected) if compare else got == expected
    if ok:
        print(f"  [PASS] {name}")
    else:
        print(
            f"  [FAIL] {name}\n         expected {expected!r}\n         got      {got!r}"
        )
    return ok


def check_raises(name, fn, exc_type, message_substring):
    try:
        fn()
    except exc_type as error:
        if message_substring in str(error):
            print(f"  [PASS] {name}")
            return True
        print(
            f"  [FAIL] {name}\n         expected message containing {message_substring!r}\n         got      {str(error)!r}"
        )
        return False
    except Exception as error:
        print(
            f"  [FAIL] {name}\n         expected {exc_type.__name__}, got {type(error).__name__}: {error}"
        )
        return False
    print(f"  [FAIL] {name}\n         expected {exc_type.__name__}, nothing raised")
    return False


def close(rel_tol):
    return lambda got, expected: math.isclose(got, expected, rel_tol=rel_tol)


def points_within_ranges(points, config):
    """True when every point is a pair of ints inside the API's allowed ranges."""
    for point in points:
        if not isinstance(point, tuple) or len(point) != 2:
            return False
        model_params, training_tokens = point
        if not isinstance(model_params, int) or not isinstance(training_tokens, int):
            return False
        if not config.min_model_params <= model_params <= config.max_model_params:
            return False
        if (
            not config.min_training_tokens
            <= training_tokens
            <= config.max_training_tokens
        ):
            return False
    return True


def plan_compute(points):
    return sum(6.0 * model_params * tokens for model_params, tokens in points)


def spans_at_least(values, factor):
    return max(values) >= factor * min(values)


def todo_1(lab):
    """TODO 1: estimate_training_compute"""
    return [
        check(
            "Test that the compute estimate equals 6*N*D",
            lambda: lab.estimate_training_compute(1_000_000, 1_000_000_000),
            6e15,
            close(1e-9),
        ),
        check(
            "Test that a model with no parameters costs no compute",
            lambda: lab.estimate_training_compute(0, 1_000_000_000),
            0.0,
            close(1e-9),
        ),
    ]


def todo_2(lab):
    """TODO 2: make_warmup_points"""
    config = lab.Config()
    try:
        points = lab.make_warmup_points()
    except NotImplementedError:
        print("  [FAIL] make_warmup_points\n         not implemented yet")
        return [False]

    varies_model = any(
        left[1] == right[1] and left[0] != right[0]
        for left in points
        for right in points
    )
    varies_data = any(
        left[0] == right[0] and left[1] != right[1]
        for left in points
        for right in points
    )
    return [
        check(
            "Test that the warm-up plan has 4 or 5 points",
            lambda: 4 <= len(points) <= 5,
            True,
        ),
        check(
            "Test that every point is an (int, int) tuple with N and D inside the Config ranges",
            lambda: points_within_ranges(points, config),
            True,
        ),
        check(
            "Test that the total cost of all points (sum of 6*N*D) fits inside the warm-up compute budget",
            lambda: plan_compute(points) <= WARMUP_COMPUTE_BUDGET,
            True,
        ),
        check(
            "Test that at least two points have the same token count and different model sizes",
            lambda: varies_model,
            True,
        ),
        check(
            "Test that at least two points have the same model size and different token counts",
            lambda: varies_data,
            True,
        ),
    ]


def todo_3(lab):
    """TODO 3: make_model_sweep"""
    config = lab.Config()
    try:
        points = lab.make_model_sweep(config)
    except NotImplementedError:
        print("  [FAIL] make_model_sweep\n         not implemented yet")
        return [False]

    model_sizes = [model_params for model_params, _ in points]
    token_counts = {tokens for _, tokens in points}
    return [
        check(
            "Test that the sweep has at least config.num_sweep_points different model sizes",
            lambda: len(set(model_sizes)) >= MIN_SWEEP_STEPS,
            True,
        ),
        check(
            "Test that every point is an (int, int) tuple with N and D inside the Config ranges",
            lambda: points_within_ranges(points, config),
            True,
        ),
        check(
            "Test that every point uses D = config.fixed_tokens_for_model_sweep",
            lambda: token_counts == {config.fixed_tokens_for_model_sweep},
            True,
        ),
        check(
            f"Test that the largest model size is at least {MIN_SWEEP_SPAN}x the smallest model size",
            lambda: spans_at_least(model_sizes, MIN_SWEEP_SPAN),
            True,
        ),
    ]


def todo_4(lab):
    """TODO 4: make_data_sweep"""
    config = lab.Config()
    try:
        points = lab.make_data_sweep(config)
    except NotImplementedError:
        print("  [FAIL] make_data_sweep\n         not implemented yet")
        return [False]

    token_counts = [tokens for _, tokens in points]
    model_sizes = {model_params for model_params, _ in points}
    return [
        check(
            "Test that the sweep has at least config.num_sweep_points different token counts",
            lambda: len(set(token_counts)) >= MIN_SWEEP_STEPS,
            True,
        ),
        check(
            "Test that every point is an (int, int) tuple with N and D inside the Config ranges",
            lambda: points_within_ranges(points, config),
            True,
        ),
        check(
            "Test that every point uses N = config.fixed_params_for_data_sweep",
            lambda: model_sizes == {config.fixed_params_for_data_sweep},
            True,
        ),
        check(
            f"Test that the largest token count is at least {MIN_SWEEP_SPAN}x the smallest token count",
            lambda: spans_at_least(token_counts, MIN_SWEEP_SPAN),
            True,
        ),
    ]


def todo_6(lab):
    """TODO 6: make_budget_experiment_plan"""
    config = lab.Config()
    budget = 1e22
    try:
        points = lab.make_budget_experiment_plan(config, budget)
    except NotImplementedError:
        print("  [FAIL] make_budget_experiment_plan\n         not implemented yet")
        return [False]

    return [
        check(
            "Test that the plan has at least MIN_BUDGET_POINTS points",
            lambda: len(points) >= lab.MIN_BUDGET_POINTS,
            True,
        ),
        check(
            "Test that every point is an (int, int) tuple with N and D inside the Config ranges",
            lambda: points_within_ranges(points, config),
            True,
        ),
        check(
            "Test that the total cost of the plan (sum of 6*N*D) fits inside the compute budget",
            lambda: plan_compute(points) <= budget,
            True,
        ),
        check(
            "Test that the total cost of the plan is at least half of the compute budget",
            lambda: plan_compute(points) >= 0.5 * budget,
            True,
        ),
        check(
            "Test that the plan uses several different model sizes",
            lambda: len({model_params for model_params, _ in points}) >= 4,
            True,
        ),
        check(
            "Test that the plan uses several different token counts",
            lambda: len({tokens for _, tokens in points}) >= 4,
            True,
        ),
    ]


def todo_7(lab):
    """TODO 7: compute_scaling_law_loss"""
    import numpy as np

    def on_arrays():
        got = lab.compute_scaling_law_loss(
            np.array([1e9, 1e10]),
            np.array([1e11, 1e12]),
            1.5,
            300.0,
            350.0,
            0.30,
            0.25,
        )
        return [float(value) for value in got]

    return [
        check(
            "Test that a point at mid-scale gets the correct loss prediction",
            lambda: lab.compute_scaling_law_loss(
                1e9, 1e11, 1.5, 300.0, 350.0, 0.30, 0.25
            ),
            2.720976488004287,
            close(1e-5),
        ),
        check(
            "Test that a point at huge scale gets a loss just above the irreducible loss",
            lambda: lab.compute_scaling_law_loss(
                1e30, 1e30, 1.5, 300.0, 350.0, 0.30, 0.25
            ),
            1.5000113679718106,
            close(1e-9),
        ),
        check(
            "Test that zero model and data coefficients return the irreducible loss",
            lambda: lab.compute_scaling_law_loss(1e9, 1e11, 1.5, 0.0, 0.0, 0.30, 0.25),
            1.5,
            close(1e-9),
        ),
    ]


def todo_8(lab):
    """TODO 8: fit_scaling_law"""
    try:
        fit = lab.fit_scaling_law(NOISE_FREE_RUNS)
    except NotImplementedError:
        print("  [FAIL] fit_scaling_law\n         not implemented yet")
        return [False]

    def worst_residual():
        return max(
            abs(
                lab.predict_loss(fit, run["model_params"], run["training_tokens"])
                - run["validation_loss"]
            )
            for run in NOISE_FREE_RUNS
        )

    return [
        check(
            "Test that the returned dict has exactly the 5 parameter names as keys",
            lambda: sorted(fit),
            sorted(DUMMY_FIT),
        ),
        check(
            "Test that the fit on noise-free runs recovers the true irreducible_loss",
            lambda: fit["irreducible_loss"],
            1.5,
            close(1e-3),
        ),
        check(
            "Test that the fit on noise-free runs recovers the true model_coefficient",
            lambda: fit["model_coefficient"],
            300.0,
            close(1e-3),
        ),
        check(
            "Test that the fit on noise-free runs recovers the true data_coefficient",
            lambda: fit["data_coefficient"],
            350.0,
            close(1e-3),
        ),
        check(
            "Test that the fit on noise-free runs recovers the true model_exponent",
            lambda: fit["model_exponent"],
            0.30,
            close(1e-3),
        ),
        check(
            "Test that the fit on noise-free runs recovers the true data_exponent",
            lambda: fit["data_exponent"],
            0.25,
            close(1e-3),
        ),
        check(
            "Test that the fitted law reproduces the measured loss of every input run",
            lambda: worst_residual() < 1e-4,
            True,
        ),
    ]


def todo_9(lab):
    """TODO 9: find_compute_optimal_point"""
    config = lab.Config()
    try:
        recommendation = lab.find_compute_optimal_point(DUMMY_FIT, 1e23, config)
    except NotImplementedError:
        print("  [FAIL] find_compute_optimal_point\n         not implemented yet")
        return [False]

    smaller = lab.find_compute_optimal_point(DUMMY_FIT, 1e21, config)
    return [
        check(
            "Test that the returned dict has exactly the 4 documented keys",
            lambda: sorted(recommendation),
            ["compute_flops", "model_params", "predicted_loss", "training_tokens"],
        ),
        check(
            "Test that the chosen run fits inside the compute budget",
            lambda: recommendation["compute_flops"] <= 1e23,
            True,
        ),
        check(
            "Test that the chosen run uses nearly all of the compute budget",
            lambda: recommendation["compute_flops"] >= 0.9e23,
            True,
        ),
        check(
            "Test that the chosen model size is close to the compute-optimal model size",
            lambda: recommendation["model_params"],
            1.31e10,
            close(0.05),
        ),
        check(
            "Test that a smaller compute budget gives a smaller model size",
            lambda: smaller["model_params"] < recommendation["model_params"],
            True,
        ),
        check(
            "Test that a smaller compute budget gives a higher predicted loss",
            lambda: smaller["predicted_loss"] > recommendation["predicted_loss"],
            True,
        ),
    ]


TODOS = {
    1: ("estimate_training_compute", todo_1),
    2: ("make_warmup_points", todo_2),
    3: ("make_model_sweep", todo_3),
    4: ("make_data_sweep", todo_4),
    6: ("make_budget_experiment_plan", todo_6),
    7: ("compute_scaling_law_loss", todo_7),
    8: ("fit_scaling_law", todo_8),
    9: ("find_compute_optimal_point", todo_9),
}


def main():
    requested = sys.argv[1:]
    numbers = [int(n) for n in requested] if requested else sorted(TODOS)
    lab = load_lab()
    passed = total = 0
    for number in numbers:
        if number not in TODOS:
            print(f"TODO {number} has no automatic tests; answer it in README.md")
            print()
            continue
        title, runner = TODOS[number]
        print(f"TODO {number}: {title}")
        results = runner(lab)
        passed += sum(results)
        total += len(results)
        print()
    print(f"{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
