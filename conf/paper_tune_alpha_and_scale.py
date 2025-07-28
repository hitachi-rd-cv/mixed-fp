from pipelines import *
from tasks import *
from pipelines import *

def get_config():
    depth = 1000
    dtype = 'float'
    size_x = 10
    size_lambda = 10
    kwargs_method = {}
    option_dummy = dict(
        size_x=size_x,
        size_lambda=size_lambda,
        n_parent=1000,
        epsilon=1e-2,
    )
    d_configs = {}

    # d_configs.update({f'neumann': ('neumann', {'scale': scale})})
    # d_configs.update({f'unroll': ('unroll', {'scale': scale})})
    scales = [0.001, 0.01, 0.1, 1.0]
    for scale in scales:
        for a in [0., 1e-4, 1e-3, 1e-2,  1e-1, 1e0]:
            d_configs.update({f'vr_no_km(scale={scale},a={a})': ('vr', {'a': a, 'scale': scale})})

    # scheduler = 'const'
    # for lr_ in [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1]:
    #     d_configs.update({f'grazzi_{scheduler}(lr={lr_})': ('grazzi', {'lr': lr_, 'scheduler': scheduler, 'scale': scale})})
    #
    # for a in [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1]:
    #     for lr_ in [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1]:
    #         d_configs.update({f'vr_km_{scheduler}(a={a},lr={lr_})': ('vr_km', {'a': a, 'lr': lr_, 'scheduler': scheduler, 'scale': scale})})

    xlabel = 'sample'
    labels = list(d_configs.keys())
    methods = [method for method, kwargs in d_configs.values()]
    params = [kwargs for method, kwargs in d_configs.values()]

    seeds = list(range(100))

    param_dict = {
        gokart.TaskOnKart.task_family: {
            "fix_random_seed_value": 0
        },
        SampleDummyGrads.task_family: {
            'dtype': dtype,
            'option_dummy': option_dummy,
        },
        CreateDummyOracle.task_family: {
            "depth": depth,
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
        HypergradEstimationPipeline.task_family: {
            "labels": labels,
            "methods": methods,
            "params": params,
            "seeds": seeds,
            "use_dummy_oracle": True,
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
        }
    }

    return param_dict
