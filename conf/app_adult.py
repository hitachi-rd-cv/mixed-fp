from pipelines import *
from tasks import *
from pipelines import *


def get_config():
    scale_reg = 1e-1
    option_dataset = {'name_scaler': 'minmax'}
    name_dataset = 'adult_fix'
    input_dim = 14
    device = 'cuda'
    n_train = 5000
    n_val = 5000
    n_test = 5000
    batch_size = 1
    depth = 100
    name_hyper_model = "multi"
    option_model = {}
    dtype = 'float'
    kwargs_method = {}
    option_oracle = {}
    # lr = params['alpha'] / (params['gamma'] + t + 1)
    d_configs = {}
    d_configs.update({'neumann': ('neumann', {'scale': 1.0})})
    d_configs.update({'unroll': ('unroll', {'scale': 1.0})})
    d_configs.update({'vr_no_km(a=0.02)': ('vr', {'a': 0.02, 'scale': 1.0})})
    d_configs.update({'grazzi_const(lr=0.2)': ('grazzi', {'lr': 0.2, 'scheduler': 'const', 'scale': 1.0})})
    d_configs.update({'vr_km_const(a=0.05,lr=0.5)': ('vr_km', {'a': 0.05, 'lr': 0.5, 'scheduler': 'const', 'scale': 1.0})})
    d_configs.update({'grazzi_linear(alpha=10.0,gamma=10.0)': ('grazzi', {'gamma': 10.0, 'alpha': 10.0, 'scheduler': 'linear', 'scale': 1.0})})
    d_configs.update({'vr_km_linear(a=0.2,alpha=50.0,gamma=100.0)': ('vr_km', {'a': 0.2, 'gamma': 100.0, 'alpha': 50.0, 'scheduler': 'linear', 'scale': 1.0})})

    xlabel = 'sample'
    labels = list(d_configs.keys())
    methods = [method for method, kwargs in d_configs.values()]
    params = [kwargs for method, kwargs in d_configs.values()]

    # Bilevel optimization parameters
    num_outer_steps = 100
    lr_outer = 2e1
    lr_inner = 0.01
    num_inner_steps = 100
    seeds = list(range(5))

    param_dict = {
        gokart.TaskOnKart.task_family: {
            "fix_random_seed_value": 0
        },
        DatasetPreparation.task_family: {
            "name_dataset": name_dataset,
            "option_dataset": option_dataset,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "input_dim": input_dim,
            "dtype": dtype
        },
        ModelInitialization.task_family: {
            "name_hyper_model": name_hyper_model,
            "kwargs_hyper": {"scale_reg": scale_reg},
            "name_dataset": name_dataset,
            "input_dim": input_dim,
            "option_model": option_model,
            "device": device,
            "dtype": dtype
        },
        CreateGradientOracle.task_family: {
            "dtype": dtype,
            "input_dim": input_dim,
            "batch_size": batch_size,
            "depth": depth,
            "kwargs_method": kwargs_method,
            "option_oracle": option_oracle,
            "device": device,
        },
        BatchHyperGradEstimations.task_family: {
            "dtype": dtype,
            "depth": depth,
            "kwargs_method": kwargs_method,
        },
        MakeOutputDict.task_family: {
        },
        PlotErrors.task_family: {
            "xlabel": xlabel,
            "xscale": "linear",
            "yscale": "log",
        },
        GetGoodResults.task_family: {
            "n_methods": {
                "neumann": 1,
                "unroll": 1,
                "vr_no_km": 1,
                "grazzi_const": 1,
                "vr_km_const": 1,
                "grazzi_linear": 1,
                "vr_km_linear": 1,
            },
            "ave_over_last": int(depth / 10)
        },
        SampleHessians.task_family: {
            "dtype": dtype,
        },
        BilevelOptimizationPipeline.task_family: {
            "labels": labels,
            "methods": methods,
            "params": params,
            "num_outer_steps": num_outer_steps,
            "num_inner_steps": num_inner_steps,
            "seeds": seeds
        },
        BilevelOuterStep.task_family: {
            "lr_outer": lr_outer,
            "lr_inner": lr_inner,
            "num_inner_steps": num_inner_steps,
            "dtype": dtype,
            "device": device,
            "name_dataset": name_dataset,
            "depth": depth,
            "batch_size": batch_size
        }
    }

    return param_dict
