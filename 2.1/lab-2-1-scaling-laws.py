"""Lab 2-1: Scaling law experiments with a simulated training API.

You are a pre-training engineer with one shot at a very expensive training
run. Before you spend the budget, you have to work out how to split it
between a bigger model and more training tokens.

Rather than training anything, this lab queries a local simulated API. You
give it a model size N and a token count D, it charges your compute budget
and returns a noisy validation loss. Your job is to design the experiments,
fit a scaling law to what you measure, and use it to pick the final run.

The API lives in scaling_api.py. You do not need to read it, and it is
deliberately unreadable.

The lab is run one stage at a time, and each stage needs its own flag. Every
stage spends part of a shared compute budget, and the API remembers what you
have already spent in a state file between executions, so a wasted run stays
wasted. Pass --reset-state to start the budget over.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from scaling_api import SimulatedScalingAPI, compute_budgets

BUDGETS = compute_budgets()

MIN_FIT_RUNS = 8
MIN_BUDGET_POINTS = 16
EXPERIMENT_COMPUTE_MARGIN = 1.05


@dataclass
class Config:
    seed: int = 1234

    min_model_params: int = 50_000_000
    max_model_params: int = 20_000_000_000
    min_training_tokens: int = 1_000_000_000
    max_training_tokens: int = 10_000_000_000_000

    query_budget: int = 60
    experiment_compute_budget: float = BUDGETS["experiment"]
    final_compute_budget: float = BUDGETS["final"]
    noise_std: float = 0.015

    fixed_tokens_for_model_sweep: int = 20_000_000_000
    fixed_params_for_data_sweep: int = 100_000_000
    num_sweep_points: int = 6

    num_budget_points: int = 24
    num_final_grid_points: int = 200
    plot_path: str = "./lab_2_1_measurements.png"

    state_path: str | None = "lab_2_1_api_state.json"
    history_path: str = "lab_2_1_api_history.txt"


def main() -> None:
    config, args = parse_args()

    print("# Lab 2-1: Scaling law experiments")
    print()

    api = SimulatedScalingAPI(config)
    if args.reset_state:
        api.reset_state()
    print_state(api, args.reset_state)

    print("# API budget")
    print_budget(api)

    if args.all or args.warmup:
        print("# Warm-up API queries")
        warmup_runs = run_points(api, make_warmup_points(), config.seed, "warmup")
        print_runs("Warm-up runs", warmup_runs)
        print_remaining_experiment_budget(api)

    if args.all or args.model_sweep:
        print("# Model-size sweep")
        model_sweep_runs = run_points(
            api, make_model_sweep(config), config.seed + 100, "model_sweep"
        )
        print_runs("Varying model size with fixed data", model_sweep_runs)
        print_remaining_experiment_budget(api)

    if args.all or args.data_sweep:
        print("# Data-size sweep")
        data_sweep_runs = run_points(
            api, make_data_sweep(config), config.seed + 200, "data_sweep"
        )
        print_runs("Varying data size with fixed model", data_sweep_runs)
        print_remaining_experiment_budget(api)

    if args.all or args.budgeted_experiments_and_scaling_law_fit:
        print("# Budgeted experiment design")
        remaining_experiment_budget = api.get_budget()["remaining_compute"]
        budget_points = make_budget_experiment_plan(config, remaining_experiment_budget)
        budget_runs = run_points(api, budget_points, config.seed + 300, "budget")
        print_runs("Budgeted runs used for fitting", budget_runs)
        print_remaining_experiment_budget(api)

        print("# Fitting a scaling law")
        fit = fit_scaling_law(budget_runs)

    if args.all or args.compute_final_run:
        print("# Choosing a final run for a larger budget")
        fit = fit_scaling_law(budgeted_runs_from_state(api))
        recommendation = find_compute_optimal_point(
            fit, config.final_compute_budget, config
        )
        print_recommendation(recommendation)
        make_measurement_plots(config, api, fit, recommendation)
        print_remaining_experiment_budget(api)

    print("# Final API budget")
    print_budget(api)
    print(f"wrote API call history to {api.dump_history()}")
    print(f"API state persisted in {config.state_path}")
    print()


def parse_args() -> tuple[Config, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="forget the compute spent by earlier executions before starting",
    )

    stages = parser.add_argument_group("stages")
    stages.add_argument("--warmup", action="store_true", help="run the warm-up points")
    stages.add_argument(
        "--model-sweep", action="store_true", help="run the model-size sweep"
    )
    stages.add_argument(
        "--data-sweep", action="store_true", help="run the data-size sweep"
    )
    stages.add_argument(
        "--budgeted-experiments-and-scaling-law-fit",
        action="store_true",
        help="run the budgeted measurements and fit the scaling law to them",
    )
    stages.add_argument(
        "--compute-final-run",
        action="store_true",
        help="refit from the budgeted runs and recommend a final large run",
    )
    stages.add_argument("--all", action="store_true", help="run every stage in order")
    args = parser.parse_args()

    stage_flags = (
        args.warmup,
        args.model_sweep,
        args.data_sweep,
        args.budgeted_experiments_and_scaling_law_fit,
        args.compute_final_run,
        args.all,
    )
    if not any(stage_flags):
        parser.error(
            "pick at least one stage: --warmup, --model-sweep, --data-sweep, "
            "--budgeted-experiments-and-scaling-law-fit, --compute-final-run, --all"
        )

    return Config(seed=args.seed), args


def budgeted_runs_from_state(api: SimulatedScalingAPI) -> list[dict]:
    """The budgeted measurements, reloaded from the persisted API state."""
    runs = [run for run in api.list_runs() if run["stage"] == "budget"]
    if not runs:
        print("No budgeted runs found in the API state.")
        print("Run the script with --budgeted-experiments-and-scaling-law-fit first.")
        raise SystemExit(3)
    return runs


def make_warmup_points() -> list[tuple[int, int]]:
    """Choose the warm-up points to send to the API.

    A point is a (model_params, training_tokens) pair. The warm-up is about
    intuition, not about fitting anything: change one of the two quantities
    while holding the other fixed, and look at what happens to the loss.

    Play around with between 4 and 5 points, every point must lie inside the
    allowed ranges (see Config), and the plan must cost at most 5e21 FLOPs in total.

    Returns:
        A list of (model_params, training_tokens) pairs, both integers.
    """

    points = [
        (3_000_000_000, 3_000_000_000),   # budget is 5.4e19
        (3_000_000_000, 10_000_000_000),  # budget is 1.8e20
        (5_000_000_000, 6_000_000_000),   # budget is 1.8e20
        (10_000_000_000, 3_000_000_000)   # budget is 1.8e20
        ]

    min_model_params = 50_000_000
    max_model_params = 20_000_000_000
    min_training_tokens = 1_000_000_000
    max_training_tokens = 10_000_000_000_000

    budget_sum = 0
    for n,d in points:
        assert min_model_params <= n <= max_model_params
        assert min_training_tokens <= d <= max_training_tokens
        budget_sum += estimate_training_compute(n,d)

    assert budget_sum <= 5e21, "plan budget exceeds limit"
    print(format_number(budget_sum))
    return points


def make_model_sweep(config: Config) -> list[tuple[int, int]]:
    """Choose the points for a one-dimensional sweep over model size.

    Every point in the sweep trains on the same number of tokens,
    `config.fixed_tokens_for_model_sweep`, so model size is the only thing
    that changes.

    The sweep must use at least `config.num_sweep_points` different model
    sizes, all inside the range Config allows, and the largest size must be
    at least 100 times the smallest.

    Args:
        config: the lab configuration, holding the allowed ranges and the
            token count every point in this sweep shares.

    Returns:
        A list of (model_params, training_tokens) pairs, both integers.
    """
    D = config.fixed_tokens_for_model_sweep
    num_sweep = config.num_sweep_points
    max_N, min_N = config.max_model_params, config.min_model_params
    lower_limit_N = min_N
    assert min_N <= lower_limit_N
    ratio_step = 3
    model_sweep_lst = np.array([lower_limit_N * pow(ratio_step, i) for i in range(num_sweep)])
    assert (model_sweep_lst < max_N).all(), "Some N value exceeds max_N"
    model_sweep_point = [(int(n),D) for n in model_sweep_lst]
    return model_sweep_point


def make_data_sweep(config: Config) -> list[tuple[int, int]]:
    """Choose the points for a one-dimensional sweep over training tokens.

    Every point in the sweep uses the same model size,
    `config.fixed_params_for_data_sweep`, so the token count is the only
    thing that changes.

    The sweep must use at least `config.num_sweep_points` different token
    counts, all inside the range Config allows, and the largest count must be
    at least 100 times the smallest.

    Args:
        config: the lab configuration, holding the allowed ranges and the
            model size every point in this sweep shares.

    Returns:
        A list of (model_params, training_tokens) pairs, both integers.
    """
    N = config.fixed_params_for_data_sweep
    num_sweep = config.num_sweep_points
    max_D, min_D = config.max_training_tokens, config.min_training_tokens
    lower_limit_D = min_D
    assert min_D <= lower_limit_D
    ratio_step = 3
    data_sweep_lst = np.array([lower_limit_D * pow(ratio_step, i) for i in range(num_sweep)])
    assert (data_sweep_lst < max_D).all(), "Some D value exceeds max_D"
    data_sweep_point = [(N, int(d)) for d in data_sweep_lst]
    return data_sweep_point


def logspace_int(start: int, stop: int, num: int) -> list[int]:
    values = np.logspace(math.log10(start), math.log10(stop), num=num)
    return sorted({int(round(value)) for value in values})


def format_number(value: float) -> str:
    return f"{value:.3e}"


def print_state(api: SimulatedScalingAPI, was_reset: bool) -> None:
    print("# API state")
    if was_reset:
        print("state was reset, so the API starts this run with a full budget")
    elif api.state_path is None:
        print("state persistence is disabled for this run")
    elif api.list_runs():
        print(
            f"resumed {len(api.list_runs())} runs from earlier executions in {api.state_path}"
        )
    else:
        print(f"no earlier state in {api.state_path}, starting from a full budget")
    print()


def print_budget(api: SimulatedScalingAPI) -> None:
    budget = api.get_budget()
    print(
        "queries:"
        f" {budget['used_queries']}/{budget['total_queries']} used,"
        f" {budget['remaining_queries']} remaining"
    )
    print(
        "compute:"
        f" {format_number(budget['used_compute'])}/{format_number(budget['total_compute'])}"
        f" FLOPs used"
    )
    print()


def print_remaining_experiment_budget(api: SimulatedScalingAPI) -> None:
    budget = api.get_budget()
    print(
        "remaining experiment compute budget:"
        f" {format_number(budget['remaining_compute'])} FLOPs"
    )
    print()


def run_points(
    api: SimulatedScalingAPI, points: list[tuple[int, int]], seed: int, stage: str
) -> list[dict]:
    runs = []
    for i, (model_params, training_tokens) in enumerate(points):
        runs.append(
            api.train_and_evaluate(
                model_params=model_params,
                training_tokens=training_tokens,
                seed=seed + i,
                stage=stage,
            )
        )
    return runs


def print_runs(title: str, runs: list[dict]) -> None:
    print(title)
    for run in runs:
        print_run(run)
    print()


def print_run(run: dict) -> None:
    print(
        f"{run['run_id']:2d}  "
        f"N={format_number(run['model_params'])}  "
        f"D={format_number(run['training_tokens'])}  "
        f"C={format_number(run['compute_flops'])}  "
        f"loss={run['validation_loss']:.4f}"
    )


def make_budget_experiment_plan(
    config: Config, remaining_compute_budget: float
) -> list[tuple[int, int]]:
    """Choose the measurements the scaling law will be fitted to.

    This is the real measurement plan, and it is the last chance to spend the
    experiment budget well. A fit is only as good as the spread of the points
    it sees, so the plan should cover several compute levels and, at each
    level, several different balances between model size and training tokens.

    Args:
        config: the lab configuration, holding the allowed ranges and
            `num_budget_points`, roughly how many measurements to plan for.
        remaining_compute_budget: the experiment compute still unspent, in
            FLOPs.

    Returns:
        A list of (model_params, training_tokens) pairs, both integers. Every
        point lies inside the ranges Config allows, and the whole list costs
        at most `remaining_compute_budget` FLOPs.
    """
    if remaining_compute_budget <= 0:
        raise ValueError("No compute remains for the budgeted fitting experiment.")

    points = []
    num_budget_points = config.num_budget_points
    max_N, min_N = config.max_model_params, config.min_model_params
    max_D, min_D = config.max_training_tokens, config.min_training_tokens

    level_num = 4
    spread_level = 6
    assert level_num * spread_level == num_budget_points, "Num of points exceeds num_budget_points"

    budget_sum = 0
    levels = np.array(logspace_int(1, 100, num=level_num)) # two decades backwards at most
    eff_remaining_budget = remaining_compute_budget * 1

    C_levels = [
        (eff_remaining_budget / (spread_level * levels.sum())) * level
         for level in levels
    ]

    for C in C_levels:
        lower_limit_N = max(min_N, C / (6 * max_D)) # lowest N could be min_N or compute / (6 * highest D)
        upper_limit_N = min(max_N, C / (6 * min_D)) # highest N could be max_N or compute / (6 * lowest D)
        Ns = logspace_int(lower_limit_N, upper_limit_N, spread_level)
        Ds = [int(round(C / (6*N))) for N in Ns]
        assert (min_D <= np.array(Ds)).all() and (np.array(Ds) <= max_D).all(), "Ds have values outside the permitted range."
        for N, D in zip(Ns, Ds):
            budget_sum += estimate_training_compute(N, D)
            points.append((N, D))

    assert budget_sum <= remaining_compute_budget, "budget exceeds available remaining budget"
    return sorted(set(points))


def compute_scaling_law_loss(
    model_params,
    training_tokens,
    irreducible_loss,
    model_coefficient,
    data_coefficient,
    model_exponent,
    data_exponent,
):
    """Predict the loss based on the Chinchilla-style scaling law.

    Args:
        model_params: N, a number or a numpy array of numbers.
        training_tokens: D, a number or a numpy array of numbers.
        irreducible_loss: E, the loss left over once N and D are both huge.
        model_coefficient: A.
        data_coefficient: B.
        model_exponent: alpha.
        data_exponent: beta.

    Returns:
        The predicted loss, shaped like `model_params` and `training_tokens`.
    """
    model_effect = model_coefficient / pow(model_params, model_exponent)
    data_effect = data_coefficient / pow(training_tokens, data_exponent)
    return model_effect + data_effect + irreducible_loss



def fitted_scaling_law(
    inputs,
    irreducible_loss,
    model_coefficient,
    data_coefficient,
    model_exponent,
    data_exponent,
):
    """compute_scaling_law_loss in the argument shape curve_fit expects."""
    model_params, training_tokens = inputs
    return compute_scaling_law_loss(
        model_params,
        training_tokens,
        irreducible_loss,
        model_coefficient,
        data_coefficient,
        model_exponent,
        data_exponent,
    )


def fit_scaling_law(runs: list[dict]) -> dict:
    """Fit the scaling law to a set of measured runs.

    Args:
        runs: the measured runs to fit, each a dict with "model_params",
            "training_tokens" and "validation_loss" keys, as the API returns.

    Returns:
        The fitted parameters, as a dict with the keys "irreducible_loss",
        "model_coefficient", "data_coefficient", "model_exponent" and
        "data_exponent", each holding a float.
    """
    if len(runs) < MIN_FIT_RUNS:
        raise ValueError(
            "Too few runs for a reliable fit. "
            f"Got {len(runs)}, but this lab expects at least {MIN_FIT_RUNS}. "
            "Run more budgeted measurements before fitting."
        )

    model_params = np.array([run["model_params"] for run in runs], dtype=float)
    training_tokens = np.array([run["training_tokens"] for run in runs], dtype=float)
    losses = np.array([run["validation_loss"] for run in runs], dtype=float)


    initial_guess = [1.0, 100.0, 100.0, 0.3, 0.3] # Parameter order: E, A, B, alpha, beta

    lower_bounds = [0.0, 1e-8, 1e-8, 0.01, 0.01]  # Parameter order: E, A, B, alpha, beta
    upper_bounds = [10.0, 1e6, 1e6, 1.0, 1.0]     # Parameter order: E, A, B, alpha, beta

    fitted_params = curve_fit(
        fitted_scaling_law,
        xdata=(model_params, training_tokens),
        ydata=losses,
        p0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        x_scale=initial_guess,
        max_nfev=20_000,
    )

    E, A, B, alpha, beta = map(float, fitted_params[0])

    return {
        "irreducible_loss":  E,
        "model_coefficient": A,
        "data_coefficient":  B,
        "model_exponent":    alpha,
        "data_exponent":     beta,
    }


def find_compute_optimal_point(
    fit: dict, compute_budget: float, config: Config
) -> dict:
    """Pick the model size and token count to spend a compute budget on.

    Args:
        fit: fitted scaling-law parameters, as `fit_scaling_law` returns them.
        compute_budget: the compute available for this one run, in FLOPs.
        config: the lab configuration, holding the allowed ranges for N and D.

    Returns:
        A dict describing the chosen run, with exactly four keys:
            "model_params": the chosen number of parameters N, an int,
            "training_tokens": the chosen number of training tokens D, an int,
            "compute_flops": what that run costs, at most `compute_budget`,
            "predicted_loss": the loss `fit` predicts for it, a float.
        Of every run that fits inside the budget, the one returned has the
        lowest predicted loss.
    """
    max_N, min_N = config.max_model_params, config.min_model_params
    max_D, min_D = config.max_training_tokens, config.min_training_tokens
    points_number = config.num_final_grid_points

    C = compute_budget
    lower_limit_N = max(min_N, C / (6 * max_D))  # lowest N could be min_N or compute / (6 * highest D)
    upper_limit_N = min(max_N, C / (6 * min_D))  # highest N could be max_N or compute / (6 * lowest D)
    Ns = logspace_int(lower_limit_N, upper_limit_N, points_number)
    Ds = [int(round(C / (6*N))) for N in Ns]
    assert (min_D <= np.array(Ds)).all() and (np.array(Ds) <= max_D).all(), "Ds have values outside the permitted range."
    loss_dict = {(N,D) : predict_loss(fit, N, D) for N,D in zip(Ns, Ds)}
    N_optim, D_optim = min(loss_dict, key=loss_dict.get)
    loss_optim = loss_dict.get((N_optim,D_optim))


    return {
        "model_params": N_optim,
        "training_tokens": D_optim,
        "compute_flops": C,
        "predicted_loss": loss_optim
    }


def predict_loss(fit: dict, model_params: int, training_tokens: int) -> float:
    return compute_scaling_law_loss(
        model_params,
        training_tokens,
        fit["irreducible_loss"],
        fit["model_coefficient"],
        fit["data_coefficient"],
        fit["model_exponent"],
        fit["data_exponent"],
    )


def print_recommendation(recommendation: dict) -> None:
    print(
        "recommended final run: "
        f"N={format_number(recommendation['model_params'])}, "
        f"D={format_number(recommendation['training_tokens'])}, "
        f"C={format_number(recommendation['compute_flops'])}, "
        f"predicted loss={recommendation['predicted_loss']:.4f}"
    )
    print()


def estimate_training_compute(model_params: int, training_tokens: int) -> float:
    """Approximate the compute needed to train a dense Transformer, in FLOPs.

    Args:
        model_params: the number of model parameters, N.
        training_tokens: the number of training tokens, D.

    Returns:
        The estimated number of floating point operations spent training a
        model with `model_params` parameters on `training_tokens` tokens.
    """
    N = model_params
    D = training_tokens
    return (6 * N * D)

def make_measurement_plots(
    config: Config,
    api: SimulatedScalingAPI,
    fit: dict,
    recommendation: dict,
) -> None:
    """Plot the measurements taken so far and the fitted extrapolation."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Plot skipped: install matplotlib to generate the measurement plots.")
        print()
        return

    model_sweep_runs = [run for run in api.list_runs() if run["stage"] == "model_sweep"]
    data_sweep_runs = [run for run in api.list_runs() if run["stage"] == "data_sweep"]
    budget_runs = [run for run in api.list_runs() if run["stage"] == "budget"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax = axes[0, 0]
    model_sizes = [run["model_params"] for run in model_sweep_runs]
    model_losses = [run["validation_loss"] for run in model_sweep_runs]
    ax.plot(model_sizes, model_losses, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("model parameters N")
    ax.set_ylabel("measured validation loss")
    ax.set_title("Noisy model-size sweep")
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[0, 1]
    token_counts = [run["training_tokens"] for run in data_sweep_runs]
    data_losses = [run["validation_loss"] for run in data_sweep_runs]
    ax.plot(token_counts, data_losses, marker="o", color="tab:green")
    ax.set_xscale("log")
    ax.set_xlabel("training tokens D")
    ax.set_ylabel("measured validation loss")
    ax.set_title("Noisy data-size sweep")
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1, 0]
    all_n = [run["model_params"] for run in budget_runs]
    all_d = [run["training_tokens"] for run in budget_runs]
    all_losses = [run["validation_loss"] for run in budget_runs]
    scatter = ax.scatter(all_n, all_d, c=all_losses, cmap="viridis_r", s=54)
    fig.colorbar(scatter, ax=ax, label="measured validation loss")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("model parameters N")
    ax.set_ylabel("training tokens D")
    ax.set_title("Budgeted measurements")

    ax = axes[1, 1]
    candidate_model_sizes = logspace_int(
        config.min_model_params,
        config.max_model_params,
        config.num_final_grid_points,
    )
    candidate_model_sizes = [
        model_params
        for model_params in candidate_model_sizes
        if config.min_training_tokens
        <= int(config.final_compute_budget / (6.0 * model_params))
        <= config.max_training_tokens
    ]
    candidate_losses = [
        predict_loss(
            fit,
            model_params,
            int(config.final_compute_budget / (6.0 * model_params)),
        )
        for model_params in candidate_model_sizes
    ]
    ax.plot(
        candidate_model_sizes,
        candidate_losses,
        color="tab:purple",
        marker="o",
        markersize=3,
        label="candidate runs",
    )
    ax.scatter(
        [recommendation["model_params"]],
        [recommendation["predicted_loss"]],
        color="black",
        s=70,
        zorder=3,
        label="recommended point",
    )
    ax.set_xscale("log")
    ax.set_xlabel(f"model parameters N at C={config.final_compute_budget:.0e} FLOPs")
    ax.set_ylabel("fitted predicted loss")
    ax.set_title("Extrapolation from fitted law")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("Lab 2-1 measurements and fitted extrapolation", fontsize=14)
    fig.savefig(config.plot_path, dpi=180)
    print(f"saved measurement plot to {config.plot_path}")
    print()


if __name__ == "__main__":
    main()
