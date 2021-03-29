---
layout: post
title:  "Some Notes on dragonfly (bayesian optimization)"
date:   2021-03-20
categories: jekyll update
mathjax: true
---

# Domain

See: [config_parser.py](https://github.com/dragonfly/dragonfly/blob/master/dragonfly/parse/config_parser.py#L137)

* "float": Needs ("min", "max"). {'type': 'float', 'min': -5.2, 'max': 5.7}

* "int": Needs ("min", "max"). {'type': 'int', 'min': -5, 'max': 3}

* "discrete": Needs [list of items]. {'type': 'discrete', 'items': ['a', 'b', 'c']}

* "discrete_numeric": Needs [list of items] or [range of items]. {'type': 'discrete_numeric', 'items': [1, 2, 3]} or {'type': 'discrete_numeric', 'items': '0.0:0.25:1'} where items are "startpoint:stepsize:endpoint". Note if passing explicit list, items need to be numeric.

* "boolean": {'type': 'boolean'}


# Config file options


List of fields from example config file:

--build_new_model_every 17
Train new GP every VALUE iterations

--capital_type realtime
Type of capital (realtime indicates time)

--is_multi_objective 1
Boolean - 1 if problem multi-objective, 0 otherwise

--max_capital 100
Capital (time budget or number of evaluations)

--max_or_min min
minimization or maximization problem

--opt_method ea
TYPE can be bo, rand, ga, ea, direct, pdoo

--report_results_every 13
Frequency of reporting progress

All options listed in:[exd_core.py](https://github.com/dragonfly/dragonfly/blob/master/dragonfly/exd/exd_core.py#L25) (grep for "get_option_specs" which is defined [here](https://github.com/dragonfly/dragonfly/blob/master/dragonfly/utils/option_handler.py#L24))

build_new_model_every: train new GP every N iterations

capital_type: Should be one of return_value, cputime, or realtime

fidel_init_method: Method to obtain initial fidels. Is used if get_initial_qinfos is None.

get_initial_qinfos: A function to obtain initial qinfos.

init_capital: The capital to be used for initialisation.

init_capital_frac: The fraction of the total capital to be used for initialisation.

init_method: Method to obtain initial queries. Is used if get_initial_qinfos is None.

init_set_to_fidel_to_opt_with_prob: 'Method to obtain initial fidels. Is used if get_initial_qinfos is None.

max_num_steps: If exceeds this many evaluations, stop.

mode: If 'syn', uses synchronous parallelisation, else asynchronous.

num_init_evals: The number of evaluations for initialisation. If <0, will use default.

prev_evaluations: Data for any previous evaluations.

progress_load_from: Load progress (from possibly a previous run) from this file.

progress_load_from_and_save_to: Load progress (from possibly a previous run) from and save results to this file. Overrides the progress_save_to and progress_load_from options.

progress_report_on_each_save: If true, will report each time results are saved.

progress_save_every: Save progress to progress_save_to every progress_save_every iterations.

progress_save_to: Save progress to this file.

report_model_on_each_build: If True, will report the model each time it is built.

report_results_every: Report results every this many iterations.
