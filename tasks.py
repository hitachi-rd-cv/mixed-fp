import os

import gokart
import numpy as np
from matplotlib import pyplot as plt
from torch import optim as optim
from tqdm import tqdm
from torch.utils.data import random_split
import torch
from lib_common import waluigi

from libs import (BinaryAdultDataset, LogisticRegression,
                  RegressionCaliforniaHousing, LinearRegression,
                  MultiLogisticRegression, HyperSingleRegularization, HyperInstanceLossWeight, HyperRepresentation,
                  GradientOracle, DummyOracle, HyperMultiRegularization, HyperMetaLearning,
                  IndexDataset, MyFashionMNIST, estimate_hypergradient)
from constants import D_ORIGINAL_DATASET_SIZE

class DatasetPreparation(waluigi.Task):
    name_dataset: str = waluigi.Parameter()
    option_dataset: dict = waluigi.DictParameter()
    n_train: int = waluigi.IntParameter()
    n_val: int = waluigi.IntParameter()
    n_test: int = waluigi.IntParameter(default=0)
    input_dim: int = waluigi.IntParameter()
    dtype: str = waluigi.Parameter()
    _ver: int = waluigi.IntParameter(6)

    def run(self):
        self.set_default_dtype(self.dtype)
        # Dataset selection
        if self.name_dataset == 'adult_fix':
            dataset_base = BinaryAdultDataset(**self.option_dataset)
        elif self.name_dataset == 'fashion_mnist':
            dataset_base = MyFashionMNIST(**self.option_dataset)
        elif self.name_dataset == 'california_regression':
            dataset_base = RegressionCaliforniaHousing(**self.option_dataset)
            assert self.input_dim == dataset_base.num_features, f'input_dim must be {dataset_base.num_features}'
        else:
            raise ValueError(f'Unknown dataset: {self.name_dataset}')

        # Split
        if self.n_test == 0 and (self.n_train, self.n_val) == D_ORIGINAL_DATASET_SIZE[self.name_dataset]:
            dataset_train = dataset_base
            if self.name_dataset == 'adult_fix':
                dataset_val = BinaryAdultDataset(train=False)
            elif self.name_dataset == 'fashion_mnist':
                dataset_val = MyFashionMNIST(train=False, **self.option_dataset)
            elif self.name_dataset == 'california_regression':
                dataset_val = RegressionCaliforniaHousing(train=False)
                assert self.input_dim == dataset_base.num_features, f'input_dim must be {dataset_base.num_features}'
            else:
                raise ValueError(f'Unknown dataset: {self.name_dataset}')
            dataset_test = None
        else:
            splits = [self.n_train, self.n_val, self.n_test, len(dataset_base) - self.n_train - self.n_val - self.n_test]
            dataset_train, dataset_val, dataset_test, _ = random_split(
                dataset_base,
                splits
            )
        self.dump((dataset_train, dataset_val, dataset_test))

class ModelInitialization(waluigi.Task):
    name_hyper_model: str = waluigi.Parameter()
    kwargs_hyper: dict = waluigi.DictParameter()
    name_dataset: str = waluigi.Parameter()
    input_dim: int = waluigi.IntParameter()
    option_model: dict = waluigi.OptionalDictParameter()
    device: str = waluigi.Parameter()
    dataset_task = gokart.TaskInstanceParameter()
    dtype: str = waluigi.Parameter()
    _ver: int = waluigi.IntParameter(7)

    def requires(self):
        return self.dataset_task

    def run(self):
        self.set_default_dtype(self.dtype)
        dataset_train, dataset_val, dataset_test = self.load()  # test set is now available
        if self.name_hyper_model == 'representation':
            input_dim_tmp = self.kwargs_hyper['rep_dim']
        else:
            input_dim_tmp = self.input_dim
        option_model = self.option_model or {}
        if self.name_dataset in ['adult_fix']:
            model = LogisticRegression(input_dim_tmp, **option_model).to(self.device)
        elif self.name_dataset in ['fashion_mnist']:
            model = MultiLogisticRegression(input_dim_tmp, 10, **option_model).to(self.device)
        elif self.name_dataset == 'california_regression':
            model = LinearRegression(input_dim_tmp, output_dim=1, **option_model).to(self.device)
        else:
            raise ValueError(f'Unknown dataset: {self.name_dataset}')
        if self.name_hyper_model == 'single':
            bilevel_model = HyperSingleRegularization(model, **self.kwargs_hyper).to(self.device)
        elif self.name_hyper_model == 'multi':
            bilevel_model = HyperMultiRegularization(model, **self.kwargs_hyper).to(self.device)
        elif self.name_hyper_model == 'influence':
            bilevel_model = HyperInstanceLossWeight(model, n_train=len(dataset_train), **self.kwargs_hyper).to(self.device)
        elif self.name_hyper_model == 'representation':
            bilevel_model = HyperRepresentation(model, input_dim=self.input_dim, **self.kwargs_hyper).to(self.device)
        elif self.name_hyper_model == 'meta':
            bilevel_model = HyperMetaLearning(model, **self.kwargs_hyper).to(self.device)
        else:
            raise ValueError(f'Unknown outer model: {self.name_hyper_model}')
        self.dump(bilevel_model)

class FullBatchInnerOptimization(waluigi.Task):
    lr = waluigi.FloatParameter()
    num_epochs = waluigi.IntParameter()
    name_dataset: str = waluigi.Parameter()
    input_dim: int = waluigi.IntParameter()
    device: str = waluigi.Parameter()
    model_task = gokart.TaskInstanceParameter()
    dataset_task = gokart.TaskInstanceParameter()
    dtype: str = waluigi.Parameter()
    _ver: int = waluigi.IntParameter(2)

    def requires(self):
        return {'model': self.model_task, 'dataset': self.dataset_task}

    def run(self):
        self.set_default_dtype(self.dtype)
        bilevel_model = self.input()['model'].load()
        dataset_train, dataset_val, dataset_test = self.input()['dataset'].load()  # test set is now available
        trained_model = self.run_in_sacred_experiment(
            self.main,
            bilevel_model=bilevel_model,
            dataset_train=dataset_train,
            device=self.device,
            lr=self.lr,
            num_epochs=self.num_epochs
        )
        self.dump(trained_model)

    @staticmethod
    def main(bilevel_model, dataset_train, device, lr, num_epochs, _run=None):
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        from torch import optim
        from libs import IndexDataset, HyperInstanceLossWeight
        train_loader = DataLoader(IndexDataset(len(dataset_train)), batch_size=len(dataset_train), shuffle=True)
        optimizer = optim.Adam(bilevel_model.model.parameters(), lr=lr)
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            bilevel_model.train()
            for indices in train_loader:
                data, target = dataset_train[list(indices)]
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                target = target.view(-1, 1).float()
                optimizer.zero_grad()
                if isinstance(bilevel_model, HyperInstanceLossWeight):
                    inner_loss = bilevel_model.inner_loss(data, target, indices=indices)
                else:
                    inner_loss = bilevel_model.inner_loss(data, target)
                inner_loss.backward()
                optimizer.step()
            pbar.set_description(f'Epoch {epoch} | Loss: {inner_loss.item():.4f}')
            if _run is not None:
                _run.log_scalar('loss', inner_loss.item(), epoch)
        return bilevel_model


class CreateGradientOracle(waluigi.Task):
    dtype: str = waluigi.Parameter()
    device: str = waluigi.Parameter()
    input_dim: int = waluigi.IntParameter()
    batch_size = waluigi.IntParameter()
    depth = waluigi.IntParameter()
    kwargs_method = waluigi.DictParameter()
    train_task = gokart.TaskInstanceParameter()
    dataset_task = gokart.TaskInstanceParameter()
    option_oracle: dict = waluigi.DictParameter()
    _ver: int = waluigi.IntParameter(12)

    def requires(self):
        return {'model': self.train_task, 'dataset': self.dataset_task}

    def run(self):
        self.set_default_dtype(self.dtype)
        bilevel_model = self.input()['model'].load()
        dataset_train, dataset_val, dataset_test = self.input()['dataset'].load()  # test set is now available
        oracle = self.main(bilevel_model, dataset_train, dataset_val, self.batch_size, self.depth, self.input_dim, self.device, self.option_oracle)
        self.dump(oracle)

    @staticmethod
    def main(bilevel_model, dataset_train, dataset_val,  batch_size, depth, input_dim=784, device='cuda', option=None):
        if option is None:
            option = {}
        oracle = GradientOracle(
            bilevel_model=bilevel_model,
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            batch_size=batch_size,
            input_dim=input_dim,
            device=device,
            depth=depth,
            **option
        )
        return oracle

class SampleDummyGrads(waluigi.Task):
    dtype: str = waluigi.Parameter()
    option_dummy: dict = waluigi.DictParameter()

    _ver: int = waluigi.IntParameter(3)

    def run(self):
        self.set_default_dtype(self.dtype)
        As, B, c, d = DummyOracle.sample_dummies(**self.option_dummy)
        self.dump((As, B, c, d))


class CreateDummyOracle(waluigi.Task):
    task_sample_dummy_grads = gokart.TaskInstanceParameter()
    depth: int = waluigi.IntParameter()
    _ver: int = waluigi.IntParameter(15)

    def requires(self):
        return self.task_sample_dummy_grads

    def run(self):
        As, B, c, d = self.load()
        oracle = DummyOracle(n=self.depth, As=As, B=B, c=c, d=d)
        self.dump(oracle)

class SampleHessians(waluigi.Task):
    task_oracle: gokart.TaskOnKart = gokart.TaskInstanceParameter()
    dtype: str = waluigi.Parameter()
    _ver: int = waluigi.IntParameter(2)

    def requires(self):
        return self.task_oracle

    def run(self):
        self.set_default_dtype(self.dtype)
        oracle = self.load()
        hessians = self.main(oracle)
        self.dump(hessians)

    @staticmethod
    def main(oracle):
        print('Sampling Hessians...')
        hessians = []
        for indices in tqdm(oracle.minibatch_indices_train[0]):
            hessians.append(oracle.sample_inner_jacobian(oracle.model.weight, indices))
        return hessians


class BatchHyperGradEstimations(waluigi.Task):
    dtype: str = waluigi.Parameter()
    method = waluigi.Parameter()
    depth = waluigi.IntParameter()
    kwargs_method = waluigi.DictParameter()
    oracle_tasks: list = gokart.ListTaskInstanceParameter()
    _ver: int = waluigi.IntParameter(13)  # Increment version

    def requires(self):
        return self.oracle_tasks

    def run(self):
        self.set_default_dtype(self.dtype)
        targets = self.input()
        v_errors_all = []
        for target in targets:
            oracle = self.load(target)
            _, v_errors = estimate_hypergradient(
                method=self.method,
                depth=self.depth,
                params=self.kwargs_method,
                oracle=oracle,
            )
            v_errors_all.append(v_errors)

        self.dump(np.array(v_errors_all))

class HyperGradEstimation(waluigi.Task):
    dtype: str = waluigi.Parameter()
    method = waluigi.Parameter()
    depth = waluigi.IntParameter()
    kwargs_method = waluigi.DictParameter()
    oracle_task: gokart.TaskOnKart = gokart.TaskInstanceParameter()
    _ver: int = waluigi.IntParameter(12)  # Increment version

    def run(self):
        self.set_default_dtype(self.dtype)
        oracle = self.oracle_task.output().load()
        # v_errors = self.run_in_sacred_experiment(
        _, v_errors = estimate_hypergradient(
            method=self.method,
            depth=self.depth,
            params=self.kwargs_method,
            oracle=oracle,
        )
        self.dump(v_errors)


class MakeOutputDict(waluigi.Task):
    tasks: list = gokart.ListTaskInstanceParameter()
    keys: list = waluigi.ListParameter()
    _ver: int = waluigi.IntParameter(3)

    def run(self):
        self.dump({key: value for key, value in zip(self.keys, self.load()['tasks'])})

class SampleHyperGradEstimations(waluigi.Task):
    tasks = gokart.ListTaskInstanceParameter()
    labels: list = waluigi.ListParameter()
    params: list = waluigi.ListParameter()
    methods: list = waluigi.ListParameter()
    pseudo_walltime: bool = waluigi.BoolParameter()
    _ver: int = waluigi.IntParameter(3)

    def run(self):
        v_errors_all = self.load()['tasks']
        d_result = self.main(
                v_errors_all=v_errors_all,
                labels=self.labels,
                methods=self.methods,
                params=self.params,
                pseudo_walltime=self.pseudo_walltime
        )
        self.dump(d_result)

    @staticmethod
    def main(v_errors_all, labels, methods, params, pseudo_walltime):
        assert len(labels) == len(methods) == len(params), 'labels, methods, and params must have the same length'
        d_result = {}
        for label, method, param, v_errors_seeds in zip(labels, methods, params, v_errors_all):
            if pseudo_walltime and method in ['vr', 'vr_km']:
                n_steps = v_errors_seeds.shape[1]
                half = n_steps // 2
                v_half = v_errors_seeds[:, :half]
                v_interleaved = np.repeat(v_half, 2, axis=1)
                v_errors_seeds = v_interleaved[:, :n_steps]
            mean_errors = v_errors_seeds.mean(axis=0)
            std_errors = v_errors_seeds.std(axis=0)
            d_result[label] = {'mean': mean_errors, 'std': std_errors, 'method': method, 'param': param}

        return d_result

class GetGoodResults(waluigi.Task):
    task_sample = gokart.TaskInstanceParameter()
    n_methods: dict = waluigi.DictParameter()
    ave_over_last: int = waluigi.IntParameter()
    _ver: int = waluigi.IntParameter(7)

    def run(self):
        d_result = self.load()['task_sample']
        d_result = self.main(d_result, self.n_methods, self.ave_over_last)
        self.dump(d_result)

    @staticmethod
    def main(d_result, n_methods, ave_over_last):
        # get n top results in d_result by computing the smallest last ave_over_last values
        d_result_good = {}
        for label, result in d_result.items():
            errors = result['mean'][-ave_over_last:]
            error = errors.mean()
            d_result_good[label] = error

        output = {}
        for name, n in n_methods.items():
            results = {k: v for k, v in d_result_good.items() if name in k}
            results = sorted(results.items(), key=lambda x: x[1])
            output.update({k: d_result[k] for k, _ in results[:n]})

        return output

class PlotErrors(waluigi.Task):
    task_sample = gokart.TaskInstanceParameter()
    xlabel: str = waluigi.Parameter()
    xscale: str = waluigi.Parameter('linear')
    yscale: str = waluigi.Parameter('log')
    _ver: int = waluigi.IntParameter(4)

    def run(self):
        d_result = self.load()['task_sample']

        fig = self.run_in_sacred_experiment(
                self.main,
                d_result=d_result,
                xlabel=self.xlabel,
                xscale=self.xscale,
                yscale=self.yscale,
                output_dir=self.local_temporary_directory,
        )
        self.dump(fig)

    @staticmethod
    def main(d_result, xlabel, xscale='linear', yscale='log', output_dir='.', _run=None):
        print("Settings of filtered results from GetGoodResults:")
        for k, v in d_result.items():
            print({k: (v['method'], v['param'])})

        plt.figure(figsize=(10, 7))
        for label, result in d_result.items():
            means = result['mean']
            stds = result['std']
            method = result['method']

            if xlabel == 'sample':
                if method in ('neumann_double', 'unroll_double', 'grazzi_double', 'vr_double', 'vr_km_double'):
                    x_range = [x * 2 for x in range(len(means))]
                else:
                    x_range = range(len(means))
            elif xlabel == 'time':
                if method in ( 'vr', 'vr_km', 'vr_double', 'vr_km_double'):
                    x_range = [x * 2 for x in range(len(means))]
                else:
                    x_range = range(len(means))
            elif xlabel == 'iteration':
                x_range = range(len(means))
            else:
                raise ValueError(f'Unknown xlabel: {xlabel}')

            plt.plot(x_range, means, label=label)
            plt.fill_between(x_range, means - stds, means + stds, alpha=0.2)

        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.ylabel('error')
        plt.xlabel(xlabel)
        plt.legend()
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'v_errors_plot.png')
        plt.savefig(path)
        plt.show()

        if _run:
            _run.add_artifact(path)

        # return figure
        return plt.gcf()


class BilevelOuterStep(waluigi.Task):
    step: int = waluigi.IntParameter()
    lr_outer: float = waluigi.FloatParameter()
    lr_inner: float = waluigi.FloatParameter()
    num_inner_steps: int = waluigi.IntParameter()
    method: str = waluigi.Parameter()
    kwargs_method: dict = waluigi.DictParameter()
    dtype: str = waluigi.Parameter()
    device: str = waluigi.Parameter()
    name_dataset: str = waluigi.Parameter()
    depth: int = waluigi.IntParameter()
    batch_size: int = waluigi.IntParameter()
    dataset_task = gokart.TaskInstanceParameter()
    _ver: int = waluigi.IntParameter(4)
    total_outer_steps: int = waluigi.IntParameter(significant=False, default=-1)
    inner_optimizer_type: str = waluigi.Parameter(default="Adam")
    inner_optimizer_kwargs: dict = waluigi.OptionalDictParameter(default={})
    outer_optimizer_type: str = waluigi.Parameter(default="SGD")
    outer_optimizer_kwargs: dict = waluigi.OptionalDictParameter(default={})
    
    def requires(self):
        if self.step == 0:
            # Initial model training: use the provided dataset_task
            return {
                "model_task": ModelInitialization(dataset_task=self.dataset_task),
                "dataset_task": self.dataset_task
            }
        else:
            # Depend on the previous outer step
            seed_prev = self.fix_random_seed_value + self.step * 1000 # 
            return {
                "prev_step": BilevelOuterStep(
                    step=self.step - 1,
                    method=self.method,
                    kwargs_method=self.kwargs_method,
                    dataset_task=self.dataset_task,      
                    name_dataset=self.name_dataset,
                    lr_outer=self.lr_outer,
                    lr_inner=self.lr_inner,
                    num_inner_steps=self.num_inner_steps,
                    dtype=self.dtype,
                    device=self.device,
                    depth=self.depth,
                    batch_size=self.batch_size,
                    total_outer_steps=self.total_outer_steps,
                    fix_random_seed_value=seed_prev
                ),
                "dataset_task": self.dataset_task
            }

    def run(self):
        self.set_default_dtype(self.dtype)
        device = self.device

        if self.step == 0:
            # Load initial model and data
            bilevel_model = self.input()["model_task"].load()
            dataset_train, dataset_val, dataset_test = self.input()["dataset_task"].load()  # test set is now available
            # Initialize optimizers
            inner_opt_cls = getattr(torch.optim, self.inner_optimizer_type)
            inner_opt_kwargs = self.inner_optimizer_kwargs or {}
            inner_optimizer = inner_opt_cls([bilevel_model.inner_param], lr=self.lr_inner, **inner_opt_kwargs)
            outer_opt_cls = getattr(torch.optim, self.outer_optimizer_type)
            outer_opt_kwargs = self.outer_optimizer_kwargs or {}
            outer_optimizer = outer_opt_cls([bilevel_model.outer_param], lr=self.lr_outer, **outer_opt_kwargs)
        else:
            # Load state from previous step
            prev_state = self.input()["prev_step"].load()
            bilevel_model = prev_state["bilevel_model"]
            dataset_train, dataset_val, dataset_test = self.input()["dataset_task"].load()  # test set is now available
            inner_opt_cls = getattr(torch.optim, self.inner_optimizer_type)
            inner_opt_kwargs = self.inner_optimizer_kwargs or {}
            inner_optimizer = inner_opt_cls([bilevel_model.inner_param], lr=self.lr_inner, **inner_opt_kwargs)
            outer_opt_cls = getattr(torch.optim, self.outer_optimizer_type)
            outer_opt_kwargs = self.outer_optimizer_kwargs or {}
            outer_optimizer = outer_opt_cls([bilevel_model.outer_param], lr=self.lr_outer, **outer_opt_kwargs)
            # Load optimizer states if they exist
            if "inner_optimizer_state" in prev_state:
                inner_optimizer.load_state_dict(prev_state["inner_optimizer_state"])
            if "outer_optimizer_state" in prev_state:
                outer_optimizer.load_state_dict(prev_state["outer_optimizer_state"])

        # 1. Inner optimization
        train_loader = torch.utils.data.DataLoader(IndexDataset(len(dataset_train)), batch_size=len(dataset_train), shuffle=True)
        tqdm_desc = f"Inner optimization steps [{self.step+1}/{self.total_outer_steps if self.total_outer_steps > 0 else '?' }]"
        for _ in tqdm(range(self.num_inner_steps), desc=tqdm_desc):
            for indices in train_loader:
                data, target = dataset_train[list(indices)]
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                target = target.view(-1, 1).float()
                inner_optimizer.zero_grad()
                if isinstance(bilevel_model, HyperInstanceLossWeight):
                    loss = bilevel_model.inner_loss(data, target, indices=indices)
                else:
                    loss = bilevel_model.inner_loss(data, target)
                loss.backward()
                inner_optimizer.step()

        # Compute for train, val, test
        def eval_on_dataset(dataset):
            loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
            data, target = next(iter(loader))
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            target = target.view(-1, 1).float()
            with torch.no_grad():
                loss = bilevel_model.outer_loss(data, target).item()
                metric = bilevel_model.metric(data, target)
            return loss, metric

        train_loss, train_metric = eval_on_dataset(dataset_train)
        val_loss, val_metric = eval_on_dataset(dataset_val)
        if dataset_test is None:
            test_loss, test_metric = (None, None)
        else:
            test_loss, test_metric = eval_on_dataset(dataset_test)

        # print results
        print(f"Step {self.step}:")
        print(f"Train loss: {train_loss:.3f}, Train metric: {train_metric:.3f}")
        print(f"Val loss: {val_loss:.3f}, Val metric: {val_metric:.3f}")
        if dataset_test is not None:
            print(f"Test loss: {test_loss:.3f}, Test metric: {test_metric:.3f}")

        # Accumulate lists of losses and metrics over outer steps
        if self.step == 0:
            train_losses = [train_loss]
            val_losses = [val_loss]
            test_losses = [test_loss]
            train_metrics = [train_metric]
            val_metrics = [val_metric]
            test_metrics = [test_metric]
        else:
            train_losses = prev_state.get("train_losses", []) + [train_loss]
            val_losses = prev_state.get("val_losses", []) + [val_loss]
            test_losses = prev_state.get("test_losses", []) + [test_loss]
            train_metrics = prev_state.get("train_metrics", []) + [train_metric]
            val_metrics = prev_state.get("val_metrics", []) + [val_metric]
            test_metrics = prev_state.get("test_metrics", []) + [test_metric]

        # 2. Create oracle for hypergradient computation using GradientOracle directly
        oracle_obj = GradientOracle(
            bilevel_model=bilevel_model,
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            batch_size=self.batch_size,
            input_dim=data.shape[1],
            device=self.device,
            depth=self.depth,
            compute_v_true=False,
        )

        # 3. Compute hypergradient
        v, _ = estimate_hypergradient(
            oracle=oracle_obj,
            depth=self.depth,
            method=self.method,
            params=self.kwargs_method,
            compute_v_true=False,
            # silent=True
        )

        # 4. Outer parameter update
        outer_optimizer.zero_grad()
        bilevel_model.outer_param.grad = v.detach()
        outer_optimizer.step()

        # Save state for the next step, including metrics
        self.dump({
            "bilevel_model": bilevel_model,
            "inner_optimizer_state": inner_optimizer.state_dict(),
            "outer_optimizer_state": outer_optimizer.state_dict(),
            "step": self.step,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "test_losses": test_losses,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        })

class SampleOuterStepResults(waluigi.Task):
    tasks = gokart.ListTaskInstanceParameter()
    label: str = waluigi.Parameter()
    method: str = waluigi.Parameter()
    params: dict = waluigi.DictParameter()
    _ver: int = waluigi.IntParameter(3)

    def run(self):
        # Each task is a BilevelOuterStep for a different seed
        results = self.load()['tasks']
        d_result = self.main(results, self.label, self.method, self.params)
        self.dump(d_result)

    @staticmethod
    def main(results, label, method, params):
        # results: list of dicts, each from a BilevelOuterStep (one per seed)
        keys = [
            "train_losses", "val_losses", "test_losses",
            "train_metrics", "val_metrics", "test_metrics"
        ]
        agg = {}
        for key in keys:
            values = [r.get(key) for r in results if r.get(key) is not None]
            if values:
                print(key)
                arr = np.array(values)
                agg[key + "_mean"] = arr.mean(axis=0)
                agg[key + "_std"] = arr.std(axis=0)
            else:
                agg[key + "_mean"] = None
                agg[key + "_std"] = None
        agg["method"] = method
        agg["params"] = params
        return {label: agg}
