# Lab 2-1: Scaling laws

## Goal

In this lab (and in the following labs) you are put in the shoes of a pre-training engineer in a large AI lab.
Your goal is to train a ridiculously large model, which is extremely expensive (and if done wrong might cause the lab to go bankrupt), so you have a single attempt at doing so!
Concretely, you are given a fixed compute budget (measured in FLOPS - Floating point operations per second) for training. 
What is not fixed is how you spend it: a bigger model trained on fewer tokens, or a smaller model trained on more tokens.
You want the lowest validation loss you can get out of that budget, i.e. a compute-optimal model.

To your help come scaling laws: formulas that predict loss from model size N and token count D, fitted on small runs and then extrapolated to bigger runs.
In this assignment, you will construct a scaling law to estimate the compute-optimal model size and token amount.

First, read about scaling laws (BEFORE class). We highly recommend the [following overview of scaling laws](https://lilianweng.github.io/posts/2026-06-24-scaling-laws).
Alternatively, you can read the [Chinchilla-style scaling laws paper](https://arxiv.org/abs/2203.15556), on which the lab will be based.

Rather than training the models yourself, and to avoid wasting too much electricity for a course lab, you will query a scaling API with a pair (N,D) of N model parameters and D training tokens.
The API returns a noisy validation loss, charging your experiment budget for the privilege.

By the end you will have designed your own measurements, fitted a Chinchilla-style scaling law to them, and used it to pick the model size and token count for the final run.
NOTE: The API keeps its state on disk between executions, so a query you ran an hour ago still counts. And the experiment budget covers **all** of your queries together, so don't spend it carelessly!


## Setup

You are given these files:

- `agents.md` — a guidelines file for coding agents. Keep it included with the code at all times.
- `README.md` - this file, containing instructions and questions
- `lab-2-1-scaling-laws.py` — the lab. Edit the code in this file.
- `scaling_api.py` — the simulated training API. You do not need to open it, nor read it.
- `test_2_1.py` — basic checks for your implementations.

To run the tests:

```
python test_2_1.py 1        # check TODO 1
python test_2_1.py 3 4      # check several TODOs at once
python test_2_1.py          # check everything
```

The lab itself runs one stage at a time, and each stage needs its own flag:

```
python lab-2-1-scaling-laws.py --warmup
python lab-2-1-scaling-laws.py --model-sweep
python lab-2-1-scaling-laws.py --data-sweep
python lab-2-1-scaling-laws.py --budgeted-experiments-and-scaling-law-fit
python lab-2-1-scaling-laws.py --compute-final-run
```

Each stage prints how much **experiment** compute you have left. That number carries over between executions. 
**If you need to start over, use the `--reset-state` argument**. But note that this throws away every measurement you have made so far, and the later stages fit on what you measure in earlier stages.

Run the stages in order, and only when the TODOs they depend on are done. Everything runs in seconds on a CPU; no dataset and no GPU are needed.

## TODOs

The TODOs are marked `# TODO COMPLETE (N)` in `lab-2-1-scaling-laws.py`. Each one is specified by the docstring of the function it sits in. Do them in order.

### TODO 1: Estimating training compute

Implement `estimate_training_compute`, returning the rough estimate for training a dense Transformer. Take into consideration the total amount of FLOPS for a forward pass and a backward pass on each of the D tokens.
You will use this function later to compute the cost of every experiment plan before you send it to the API.

* Hint: Each multiply-add operation requires 2 FLOPS (one for multiply, one for add).

* Hint #2: A single backward pass computes gradients w.r.t the activations and the weights, yielding roughly twice the needed compute compared to a single forward pass.


```
python test_2_1.py 1
```

### TODO 2: Warm-up measurements

Before we start getting some larger scale training runs in you should get some hands-on feeling on how the loss behaves when we change the number of parameters N and training tokens D.
Implement `make_warmup_points`, where you define the (N,D) pairs for several smaller models. 
Before picking the (N,D) pairs, make some hypotheses on the expected loss behavior (e.g. what will happen when the total compute budget stays the same, but you give more budget through D or through N?).
Pick some (N,D) pairs that will answer your hypotheses, and give you an *intuition* on how changing N or D affects the loss. 

After implementing, run the tests to make sure your run is within the compute limits:

```
python test_2_1.py 2
```

And then train the warmup runs:
```
python lab-2-1-scaling-laws.py --warmup
```

**Question:** What did the warm-up runs tell you? Were your hypotheses validated? Was there anything in the numbers you did not expect?

**Answer:**
Our initial dual hypotheses were: 
- Increasing overall budget by 3 would reduce our loss regardless of how the budget was spent (more capacity vs more data)
- Smaller models with more data would outperform increasingly complex models on a few data given fixed budget, 
  as in Chinchilla paper observation : Instead of “train a big model and stop before convergence,” you should train a smaller model on more data"

Our experiment results are : 
""""

 1  N=3.000e+09  D=3.000e+09  C=5.400e+19  loss=2.8359

 2  N=3.000e+09  D=1.000e+10  C=1.800e+20  loss=2.5745

 3  N=5.000e+09  D=6.000e+09  C=1.800e+20  loss=2.6691

 4  N=1.000e+10  D=3.000e+09  C=1.800e+20  loss=2.7559

""""

These results support our initial hypotheses : 
- First, we saw that increased overall budget resulted in decreased loss (2.8359 -> 2.5745)
- We checked 3 configurations where budget is uniformly bigger than first configuration - C -> 3C :
  - N stayed the same as experiment #1 while D increased by a ratio of ~3
  - N increased by a ratio of 1.5~ while D increased by a ratio of ~2
  - N increased by a ratio of ~3 while D stayed the same as experiment #1

The configuration in experiment #2 outperformed other configuration supporting our second hypothesis.


### TODO 3: Sweeping model size

The warm-up gave you a few scattered points. 

To see how the loss depends on one variable (D or N), you need to hold the other parameter fixed. 
Now let's try to isolate the effect of model size N, by fixing the number of tokens to `config.fixed_tokens_for_model_sweep`.

You have been given the limits of the model size for this sweep (`config.min_model_params` and `config.max_model_params`). 
Within this range, think of a good set of training points (at least `config.num_sweep_points` runs) to check different model sizes. 

Before you pick the sizes, discuss and hypothesize the results. For example, do you think each 10x increase in N buy the same loss improvement? Will the improvement ever stop? If so, when? (No need to write down these answers).

When you have some sizes in mind, implement `make_model_sweep`, and verify the implementation:

```
python test_2_1.py 3
```

After the tests pass, run the real sweep over model size. Note that `--model-sweep` runs only the sweep. It does not re-run the warm-up, as re-running a stage would spend the budget twice:
```
python lab-2-1-scaling-laws.py --model-sweep
```

**Question:** What does the loss do as the model grows with the token count held fixed? Does it keep improving at the same rate?

**Answer:**
The loss shows persistent decreasing trend as model grows in parameters.
We started with a model size of 50_000_000, and we increased by a chosen ratio of 3 each iteration.
We saw the greatest improvement by the second iteration (1.5e8 -> 4.5e8) with 0.16~ loss decrease.
Subsequent iterations showed decreasing trend but at a slower rate (0.13~, 0.11~ and 0.05~) as model size increased.
As expected as model size grows, marginal loss improvement would decrease, and we might see an increase in the loss as 
model could potentially overfit on the data.

### TODO 4: Sweeping data size

Now let's do the same for the number of tokens D, by fixing the model size to `config.fixed_params_for_data_sweep`.

You have been given the limits of the token count for this sweep (`config.min_training_tokens` and `config.max_training_tokens`).
Within this range, think of a good set of training points (at least `config.num_sweep_points` runs) to check different token counts.
Keep in mind that every run is charged to your experiment budget, and that the cost of a run grows with D.

Before you pick the token counts, discuss and hypothesize the results. For example, does more data keep helping a fixed-size model? Do you expect the same behavior you saw when you grew N? (No need to write down these answers).

When you have some token counts in mind, implement `make_data_sweep`, and verify the implementation:

```
python test_2_1.py 4
```

After the tests pass, run the real sweep over token count:
```
python lab-2-1-scaling-laws.py --data-sweep
```

**Question:** What does the loss do as the token count grows with the model held fixed? How does this compare with what you saw in TODO 3?

**Answer:**
We started with a toke count of 1B, and we increased by a chosen ratio of 3 each iteration.
Loss remains approximately constant around 2.84–2.87 from 1B to 81B training tokens, then decreases more noticeably to
2.74 at 243B tokens. 
Compared with the model-size sweep, which showed a smoother loss decrease from 2.86 to 2.38, this data-size sweep shows weaker improvement over the tested range. 
Thus, increasing the token count helps overall, but the measured improvement was weaker for smaller models.


### TODO 5: The limits of one-dimensional sweeps

Remember the goal of the lab: given a fixed compute budget (in FLOPs) for the final run, choose how to split it between N and D.

**Question:** You have now varied model size alone and token count alone. Why are these one-dimensional sweeps not enough to answer how to split a fixed compute budget between N and D?

**Answer:**
While these one-dimensional sweeps are showing us helpful trends : increase in N or D will decrease loss,
they won't tackle our main point of interest : "given a fixed compute budget, choose how to split it between N and D".
For that we need to examine how different ratios for N and D given some fixed C, over several C's, effect the loss.
Then we might infer a useful splitting rule and answer our main question.

### TODO 6: Designing the budgeted experiments

Now that you have some intuition on how N and D affect the loss, it's time to design the experiments that will actually answer the question.
These runs are the measurements we will fit the scaling law to (in TODO 8), so the quality of the fit depends on the points you choose here.

Implement `make_budget_experiment_plan`. This plan spends whatever is left of the experiment budget. 

Before you write any code, think about what the fit needs. For example, which compute levels should you measure at? How many runs per level? How should the runs within a level differ from each other? (No need to write down these answers).

We suggest you think about the two following questions: what range of compute levels to use and which set of N/D ratios to use at each compute level.

* Hint: Once you fix the compute C of a run and its model size N, the token count follows (from your computation at `estimate_training_compute`)
* Hint #2: Some functions that might be useful (depending on your implementation, you definitely don't have to use them): `estimate_training_compute` for your planning and for the function itself, `logspace_int` to spread sizes within a single compute level.
* Hint #3: Derive your implementation based on the remaining compute budget. e.g. don't use (N, D) pairs that fall outside their ranges in `config` / cost too much.

Verify your function with some sanity checks (note that these are not comprehensive - **there is no single "correct" answer at this stage**).

```
python test_2_1.py 6
```

### TODO 7: The scaling law

The next step is to use your measurements to a prediction for the final run which will give the best results given the defined final compute.
To turn measurements into a prediction, we need a formula for the loss as a function of N and D.
We assume our model loss can be predicted with Chinchilla-style scaling laws ([Hoffmann et al.](https://arxiv.org/abs/2203.15556)), because we want to make your life easier.

Implement the Chinchilla-style loss-prediction formula in `compute_scaling_law_loss`, which predicts the loss given the amount of tokens D, model size N and the fitted parameters (which you will fit in the next stage).

```
python test_2_1.py 7
```

### TODO 8: Fitting the scaling law

The formula from TODO 7 has five unknown parameters (irreducible loss, model coefficient, data coefficient, model exponent and data exponent).
With these values, you can predict the loss of runs you never ran (at scales you didn't run), including the final run.

Implement `fit_scaling_law`. You can use `scipy.optimize.curve_fit` (read [the documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)!) with `fitted_scaling_law`. 
Alternatively, you can use any other method of curve fitting which you prefer.

The function should return a dictionary containing the five Chinchilla-style scaling law parameters.

Hint: For `curve_fit`, you will have to supply some parameter ranges. The five parameters live on wildly different scales, and the objective has flat regions, so a bad starting point converges somewhere useless or not at all.
Give it a starting guess (the `p0` parameter to `curve_fit`) and a `(lower, upper)` `bounds` pair with a plausible minimum and maximum for each of the five parameters.
Sensible bounds matter more than a clever starting guess: they keep the optimizer out of regions that are not physically meaningful, such as a negative irreducible loss or an exponent above 1.

Run some sanity tests with:
```
python test_2_1.py 8
```

Once the tests pass, its time to fit the scaling law! The following command sends your budgeting plan to the API and fits your law based on your implementation of `fit_scaling_law`:
```
python lab-2-1-scaling-laws.py --budgeted-experiments-and-scaling-law-fit
```

**Question:** Look at the fitted parameters. Which of the two terms, the model term (`model_coefficient / N^model_exponent`) or the data term (`data_coefficient / D^data_exponent`), affects the loss faster? (i.e. which of the terms shrinks faster when you increase N and D)? Note that this isn't neccessarily true for ALL fitted models (it changes with model type, data content, etc).

**Answer:**
Let's take a look on the expected loss function as in kaplan and chinchilla papers : 

L(N,D) = model_coefficient / N^model_exponent + data_coefficient / D^data_exponent + irreducible_loss

Increasing N or D in 1 decade, the terms would shrink by the exponent parameter as so:

N_new = 10N -> new model term = 1/10^model_exponent * old model term

D_new = 10D -> new data term =  1/10^data_exponent * old data term

Given our fitted parameters :

"irreducible_loss":  1.640838766273987,

"model_coefficient": 256.23921228380266,

"data_coefficient":  130.1685301158447,

"model_exponent":    0.31613181144384844,

"data_exponent":     0.22936221446829738,

we can conclude that the model term would shrink faster as it's exponent parameter is bigger.

For every decade increase in N the model term would shrink by ~50%, while same increase in D would shrink the data
term by ~40%.


### TODO 9: Choosing the final run

Finally, it's time to use the fitted law for the decision the whole lab builds toward. You have a single attempt at the final run, so its (N, D) must come from the fitted law.

Implement `find_compute_optimal_point`, which returns the compute-optimal parameters (the parameters that will yield the lowest loss given the available compute).

This function receives the fitted parameters and the compute budget for a single final training run, and returns the (N, D) pair to spend it on.
The (N, D) pair must stay inside the limits given in `config`.
You may use `logspace_int`, `predict_loss`.

The fitted law is cheap to evaluate, so you can find the best pair with a simple search over model sizes:

Implementation hints:
1. Build a grid of candidate model sizes N between `config.min_model_params` and `config.max_model_params`. Use `config.num_final_grid_points` candidates.
2. For each candidate N, choose the token count D that spends the whole budget. Remember that the compute of a run is about 6ND. Skip any candidate whose D falls outside the allowed range from the config.
4. For each remaining candidate, compute its predicted loss with `predict_loss`.
5. Return the candidate with the lowest predicted loss, as a dict with the four keys the docstring describes.

After implementing the function, run some sanity tests:
```
python test_2_1.py 9
```

and run the prediction: 
```
python lab-2-1-scaling-laws.py --compute-final-run
```

This stage also writes `lab_2_1_measurements.png`, showing your measurements against your fitted law.
The bottom-right panel shows the predicted loss of every candidate (N, D) pair that spends the final budget, with the recommended point (leading to the lowest point) marked. 

Look at the graphs and verify they make sense and that you understand what appears in each of them, and that it matches your expectations.
If so: Congratulations! You have succesfully predicted the optimal setup for a scaled-up training run, such that it will lead to the lowest possible loss!


## Lab debrief
**Question:** We're interested in any thoughts you have about this lab so that we can improve this lab for later years, and to inform later labs for this year. Please list any issues that arose or comments you have to improve the lab. Useful things to comment on include the following, but you're not restricted to these:
* Do you feel like you understand the main point of scaling laws, and how to compute the (N,D) pair for large training runs?
* Were the instructions clear?
* Was it clear (at least after you completed the lab) what the points of the exercises were?

**Answer:**

## Submission

Collect all the files in this directory into a **flat** zip file (**NO** internal directories) and submit it via Gradescope.