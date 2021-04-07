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

**All options: grep for "get_option_specs" which is defined [here](https://github.com/dragonfly/dragonfly/blob/master/dragonfly/utils/option_handler.py#L24)**

acq: Which acquisition to use: ts,  ucb,  ei,  ttei,  bucb. If using multiple  

acq_opt_max_evals: Number of evaluations when maximising acquisition. If negative uses default value. 

acq_opt_method: Which optimiser to use when maximising the acquisition function. 
* options: ['direct', 'pdoo', 'rand', 'default'] [Link](https://github.com/dragonfly/dragonfly/blob/a579b5eadf452e23b07d4caf27b402703b0012b7/dragonfly/opt/gp_bandit.py#L713)
* Rules for picking opt for specific domain types - [get_default_acq_opt_method_for_domain](https://github.com/dragonfly/dragonfly/blob/a579b5eadf452e23b07d4caf27b402703b0012b7/dragonfly/opt/gp_bandit.py#L154). Used when acq_opt_method=='default'. See [here](https://github.com/dragonfly/dragonfly/blob/a579b5eadf452e23b07d4caf27b402703b0012b7/dragonfly/opt/gp_bandit.py#L279)

acq_probs: With what probability should we choose each strategy given in acq. If "uniform"  
* options: ['uniform', 'adaptive']. Can also pass...
* 'adaptive': See [here](https://github.com/dragonfly/dragonfly/blob/a579b5eadf452e23b07d4caf27b402703b0012b7/dragonfly/opt/gp_bandit.py#L253).


add_group_size_criterion: Specify how to pick the group size,  should be one of {max,  sampled}. 

add_grouping_criterion: Specify the grouping algorithm,  should be one of {randomised_ml} 

add_max_group_size: The maximum number of groups in the additive grouping.  

boca_max_low_fidel_cost_ratio: If the fidel_cost_ratio is larger than this,  just query at fidel_to_opt. 

boca_thresh_coeff_init: The coefficient to use in determining the threshold for boca. 

boca_thresh_multiplier: The amount by which to multiply/divide the threshold coeff for boca. 

boca_thresh_window_length: The window length to keep checking if the target fidel_to_opt is achieved. 

build_new_model_every: Updates the model via a suitable procedure every this many iterations. 

capital_type: Should be one of return_value,  cputime,  or realtime 

choose_mislabel_struct_coeffs: How to choose the mislabel and struct coefficients. Should be one of  

compute_kernel_from_dists: Should you compute the kernel from pairwise distances whenever possible. 

dist_type: The type of distance. This should be lp,  emd or lp-emd. 

dom_disc_hamming_use_same_weight: If true,  use same weight for all dimensions of the hamming kernel. 

dom_disc_kernel_type: Kernel type for discrete domains. 

dom_disc_num_esp_kernel_type: Specify type of kernel. This depends on the application. 

dom_disc_num_esp_matern_nu: Specify the nu value for matern kernel. If negative,  will fit. 

dom_disc_num_esp_order: Order of the esp kernel.  

dom_disc_num_kernel_type: Kernel type for discrete numeric domains.  

dom_disc_num_matern_nu: Specify nu value for matern kernel. If negative,  will fit. 

dom_disc_num_poly_order: Order of the polynomial kernle to be used for Integral domains.  

dom_disc_num_use_same_bandwidth: If true,  will use same bandwidth on all dimensions. Applies only  

dom_euc_add_group_size_criterion: Specify how to pick the group size,  should be one of {max, sampled}. 

dom_euc_add_grouping_criterion: Specify the grouping algorithm,  should be one of {randomised_ml} 

dom_euc_add_max_group_size: The maximum number of groups in the additive grouping.  

dom_euc_esp_kernel_type: Specify type of kernel. This depends on the application. 

dom_euc_esp_matern_nu: Specify the nu value for matern kernel. If negative,  will fit. 

dom_euc_esp_order: Order of the esp kernel.  

dom_euc_kernel_type: Kernel type for euclidean domains.  

dom_euc_matern_nu: Specify nu value for matern kernel. If negative,  will fit. 

dom_euc_num_groups_per_group_size: The number of groups to try per group size. 

dom_euc_poly_order: Order of the polynomial kernle to be used for Euclidean domains.  

dom_euc_use_additive_gp: Whether or not to use an additive GP.  

dom_euc_use_same_bandwidth: If true,  will use same bandwidth on all dimensions. Applies only  

dom_int_add_group_size_criterion: Specify how to pick the group size,  should be one of {max, sampled}. 

dom_int_add_grouping_criterion: Specify the grouping algorithm,  should be one of {randomised_ml} 

dom_int_add_max_group_size: The maximum number of groups in the additive grouping.  

dom_int_esp_kernel_type: Specify type of kernel. This depends on the application. 

dom_int_esp_matern_nu: Specify the nu value for matern kernel. If negative,  will fit. 

dom_int_esp_order: Order of the esp kernel.  

dom_int_kernel_type: Kernel type for integral domains.  

dom_int_matern_nu: Specify nu value for matern kernel. If negative,  will fit. 

dom_int_num_groups_per_group_size: The number of groups to try per group size. 

dom_int_poly_order: Order of the polynomial kernle to be used for Integral domains.  

dom_int_use_additive_gp: Whether or not to use an additive GP.  

dom_int_use_same_bandwidth: If true,  will use same bandwidth on all dimensions. Applies only  

dom_nn_kernel_type: Kernel type for NN Domains. 

domain_add_group_size_criterion: Specify how to pick the group size,  should be one of {max,  sampled}. 

domain_add_grouping_criterion: Specify the grouping algorithm,  should be one of {randomised_ml} 

domain_add_max_group_size: The maximum number of groups in the additive grouping.  

domain_esp_kernel_type: Specify type of kernel. This depends on the application. 

domain_esp_matern_nu: Specify the nu value for matern kernel. If negative,  will fit. 

domain_esp_order: Order of the esp kernel.  

domain_kernel_type: Type of kernel for the domain space. Should be se,  matern or poly 

domain_matern_nu: Specify the nu value for the matern kernel. If negative,  will fit. 

domain_num_groups_per_group_size: The number of groups to try per group size. 

domain_poly_order: Order of the polynomial for domainity kernel. Default = -1 (means will fit) 

domain_use_additive_gp: Whether or not to use an additive GP.  

domain_use_same_bandwidth: If true,  will use same bandwidth on all domain dimensions. Applies only when  

domain_use_same_scalings: If true,  will use same scaling on all domainity dimensions. Applies only when  

emd_power: The powers to use in the EMD distance for the kernel. 

en_masse_dec_change_frac: Default change fraction when decreasing layers en_masse. 

en_masse_inc_change_frac: Default change fraction when increasing layers en_masse. 

esp_kernel_type: Specify type of kernel. This depends on the application. 

esp_matern_nu: Specify the nu value for matern kernel. If negative,  will fit. 

esp_order: Order of the esp kernel.  

euc_init_method: Method to obtain initial queries. Is used if get_initial_qinfos is None. 

fidel_disc_hamming_use_same_weight: If true,  use same weight for all dimensions of the hamming kernel. 

fidel_disc_kernel_type: Kernel type for discrete domains. 

fidel_disc_num_kernel_type: Type of kernel for the fidelity space. Should be se,  matern,  poly or expdecay 

fidel_disc_num_matern_nu: Specify the nu value for the matern kernel. If negative,  will fit. 

fidel_disc_num_use_same_bandwidth: If true,  will use same bandwidth on all integral fidelity dimensions. Applies  

fidel_esp_kernel_type: Specify type of kernel. This depends on the application. 

fidel_esp_matern_nu: Specify the nu value for matern kernel. If negative,  will fit. 

fidel_esp_order: Order of the esp kernel.  

fidel_euc_kernel_type: Type of kernel for the Euclidean part of the fidelity space. Should be se,   

fidel_euc_matern_nu: Specify the nu value for the matern kernel. If negative,  will fit. 

fidel_euc_use_same_bandwidth: If true,  will use same bandwidth on all Euclidean fidelity dimensions. Applies  

fidel_init_method: Method to obtain initial fidels. Is used if get_initial_qinfos is None. 

fidel_int_kernel_type: Type of kernel for the fidelity space. Should be se,  matern,  poly or expdecay 

fidel_int_matern_nu: Specify the nu value for the matern kernel. If negative,  will fit. 

fidel_int_use_same_bandwidth: If true,  will use same bandwidth on all integral fidelity dimensions. Applies  

fidel_kernel_type: Type of kernel for the fidelity space. Should be se,  matern,  poly or expdecay 

fidel_matern_nu: Specify the nu value for the matern kernel. If negative,  will fit. 

fidel_poly_order: Order of the polynomial for fidelity kernel. Default = -1 (means will tune) 

fidel_use_same_bandwidth: If true,  will use same bandwidth on all fidelity dimensions. Applies only when  

fidel_use_same_scalings: If true,  will use same scaling on all fidelity dimensions. Applies only when  

fitness_sampler_scaling_const: The scaling constant for sampling according to exp_probs. 

get_initial_qinfos: A function to obtain initial qinfos. 

gpb_hp_tune_criterion: Which criterion to use when tuning hyper-parameters. Other  

gpb_hp_tune_probs: With what probability should we choose each strategy given in hp_tune_criterion. 

gpb_ml_hp_tune_opt: Which optimiser to use when maximising the tuning criterion. 

gpb_post_hp_tune_burn: How many initial samples to ignore during sampling. 

gpb_post_hp_tune_method: Which sampling to use when maximising the tuning criterion. Other  

gpb_post_hp_tune_offset: How many samples to ignore between samples. 

gpb_prior_mean: The prior mean of the GP for the model. 

handle_non_psd_kernels: How to handle kernels that are non-psd. 

handle_parallel: How to handle parallelisations. Should be halluc or naive. 

hp_tune_criterion: Which criterion to use when tuning hyper-parameters. Other  

hp_tune_max_evals: How many evaluations to use when maximising the tuning criterion. 

hp_tune_probs: With what probability should we choose each strategy given in hp_tune_criterion. 

init_capital: The capital to be used for initialisation. 

init_capital_frac: The fraction of the total capital to be used for initialisation. 

init_method: Method to obtain initial queries. Is used if get_initial_qinfos is None. 

init_set_to_fidel_to_opt_with_prob: Method to obtain initial fidels. Is used if get_initial_qinfos is None. 

kernel_type: Specify type of kernel. This depends on the application. 

lp_power: The powers to use in the LP distance for the kernel. 

matern_nu: Specify the nu value for the matern kernel. If negative,  will fit. 

max_num_steps: If exceeds this many evaluations,  stop. 

mean_func: The mean function. If not None,  will use this instead of the 

mean_func_const: The constant value to use if mean_func_type is const. 

mean_func_type: Specify the type of mean function. Should be mean,  median,  const  

mf_strategy: Which multi-fidelity strategy to use. Should be one of {boca}. 

mislabel_coeffs: The mislabel coefficients specified as a string. If -1,  it means we will tune. 

ml_hp_tune_opt: Which optimiser to use when maximising the tuning criterion. 

mode: If \ 

moo_gpb_prior_means: Prior GP mean functions for Multi-objective GP bandits. 

moo_strategy: Get name of multi-objective strategy. So far,  Dragonfly only supports moors. 

moors_reference_point: Reference point for MOORS. 

moors_scalarisation: Scalarisation for MOORS. Should be "tchebychev" or "linear". 

moors_weight_sampler: A weight sampler for moors. 

next_pt_std_thresh: If the std of the queried point queries below this times the kernel scale  

nn_report_results_every: If NN variables are present,  report results more frequently 

noise_var_label: The fraction of label variance to use as noise variance. 

noise_var_type: Specify how to obtain the noise variance. Should be tune,  label or value.  

noise_var_value: The (absolute) value to use as noise variance. 

non_assignment_penalty: The non-assignment penalty. 

num_candidates_to_mutate_from: The number of candidates to choose the mutations from. 

num_groups_per_group_size: The number of groups to try per group size. 

num_init_evals: The number of evaluations for initialisation. If <0,  will use default. 

num_mutations_per_epoch: Number of mutations per epoch. 

num_single_step_modifications: Default number of networks to spawn via single step primitives. 

num_three_step_modifications: Default number of networks to spawn via three step primitives. 

num_two_step_modifications: Default number of networks to spawn via two step primitives. 

otmann_choose_mislabel_struct_coeffs: How to choose the mislabel and struct coefficients. Should be one of  

otmann_dist_type: The type of distance. Should be lp,  emd or lp-emd. 

otmann_emd_power: The powers to use in the EMD distance for the kernel. 

otmann_kernel_type: The Otmann kernel type. Should be one of lp,  emd,  lpemd_sum,  or lpemd_prod. 

otmann_lp_power: The powers to use in the LP distance for the kernel. 

otmann_mislabel_coeffs: The mislabel coefficients specified as a string. If -1,  it means we will tune. 

otmann_non_assignment_penalty: The non-assignment penalty for the OTMANN distance. 

otmann_struct_coeffs: The struct coefficients specified as a string. If -1,  it means we will tune. 

perturb_thresh: If the next point chosen is too close to an exisiting point by this times the  

poly_order: Order of the polynomial to be used. Default is 1 (linear kernel). 

post_hp_tune_burn: How many initial samples to ignore during sampling. 

post_hp_tune_method: Which sampling to use when maximising the tuning criterion. Other  

post_hp_tune_offset: How many samples to ignore between samples. 

prev_evaluations: Data for any previous evaluations. 

progress_load_from: Load progress (from possibly a previous run) from this file. 

progress_load_from_and_save_to: Load progress (from possibly a previous run) from and save results to this file.  

progress_report_on_each_save: If true,  will report each time results are saved. 

progress_save_every: Save progress to progress_save_to every progress_save_every iterations. 

progress_save_to: Save progress to this file. 

rand_exp_sampling_replace: Whether to replace already sampled values or not in rand_exp_sampling. 

report_model_on_each_build: If True,  will report the model each time it is built. 

report_results_every: Report results every this many iterations. 

shrink_kernel_with_time: If True,  shrinks the kernel with time so that we don\ 

single_dec_change_frac: Default change fraction when decreasing a single layer. 

single_inc_change_frac: Default change fraction when increasing a single layer. 

spawn_add_layer: Default number of networks to spawn by adding a layer. 

spawn_del_layer: Default number of networks to spawn by deleting a layer. 

spawn_single_dec_num_units: Default number of networks to spawn by decreasing # units in a single layer. 

spawn_single_inc_num_units: Default number of networks to spawn by increasing # units in a single layer. 

struct_coeffs: The struct coefficients specified as a string. If -1,  it means we will tune. 

target_fidel_to_opt_query_frac_max: A target to maintain on the number of queries to fidel_to_opt. 

target_fidel_to_opt_query_frac_min: A target to maintain on the number of queries to fidel_to_opt. 

track_every_time_step: If 1,  it tracks every time step. 

use_additive_gp: Whether or not to use an additive GP.  

use_same_bandwidth: If true,  will use same bandwidth on all dimensions. Applies only  

use_same_scalings: If true uses same scalings on all dimensions. Default is False. 

