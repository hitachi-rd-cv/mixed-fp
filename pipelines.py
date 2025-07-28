import os
import matplotlib.pyplot as plt

from lib_common import waluigi
from tasks import *

class HypergradEstimationPipeline(waluigi.Task):
    labels: list = waluigi.ListParameter()
    methods: list = waluigi.ListParameter()
    params: list = waluigi.ListParameter()
    seeds: list = waluigi.ListParameter()
    use_dummy_oracle: bool = waluigi.BoolParameter()

    def requires(self):
        if self.use_dummy_oracle:
            train_task = SampleDummyGrads()
        else:
            # Chain: DatasetPreparation -> ModelInitialization -> InnerOptimization
            dataset_task = DatasetPreparation()
            model_task = ModelInitialization(dataset_task=dataset_task)
            train_task = FullBatchInnerOptimization(model_task=model_task, dataset_task=dataset_task)
        oracle_tasks = []
        for seed in self.seeds:
            if self.use_dummy_oracle:
                oracle_task = CreateDummyOracle(fix_random_seed_value=seed, task_sample_dummy_grads=train_task)
            else:
                oracle_task = CreateGradientOracle(fix_random_seed_value=seed, train_task=train_task, dataset_task=dataset_task)
            oracle_tasks.append(oracle_task)

        tasks_method = []
        for method, kwargs in zip(self.methods, self.params):
            hypergrad_task = BatchHyperGradEstimations(method=method, oracle_tasks=oracle_tasks, kwargs_method=kwargs)
            tasks_method.append(hypergrad_task)

        task_sample = SampleHyperGradEstimations(tasks=tasks_method, labels=self.labels, methods=self.methods, params=self.params)
        task_top = GetGoodResults(task_sample=task_sample)
        plot_task = PlotErrors(task_sample=task_top)
        return plot_task, task_top, task_sample


class BilevelOptimizationPipeline(waluigi.Task):
    labels: list = waluigi.ListParameter()
    methods: list = waluigi.ListParameter()
    params: list = waluigi.ListParameter()
    seeds: list = waluigi.ListParameter()
    num_outer_steps: int = waluigi.IntParameter()
    y_scale: str = waluigi.Parameter('linear')  # Options: 'linear', 'log', 'symlog'
    _ver: int = waluigi.IntParameter(6)

    def requires(self):
        self.dataset_task = DatasetPreparation()
        tasks = {}
        for label, method, kwargs in zip(self.labels, self.methods, self.params):
            outer_step_tasks = []
            for seed in self.seeds:
                outer_step_tasks.append(
                    BilevelOuterStep(
                        step=self.num_outer_steps - 1,
                        method=method,
                        kwargs_method=kwargs,
                        dataset_task=self.dataset_task,
                        total_outer_steps=self.num_outer_steps,
                        fix_random_seed_value=seed,
                    )
                )
            tasks[label] = SampleOuterStepResults(
                label=label,
                method=method,
                params=kwargs,
                tasks=outer_step_tasks,
            )
        return tasks

    def run(self):
        loaded = self.load()
        # loaded: dict[label] = {label: agg}
        # Merge all label dicts into one
        results = {}
        for label, d in loaded.items():
            results[label] = d[label]

        fig = self.run_in_sacred_experiment(
            self.main,
            results=results,
            output_dir=self.local_temporary_directory,
            y_scale=self.y_scale,
        )
        self.dump((fig, results))

    @staticmethod
    def main(results, output_dir='bilevel_metrics_figs', _run=None, y_scale='linear'):
        os.makedirs(output_dir, exist_ok=True)
        splits = ['train', 'val', 'test']
        types = ['losses', 'metrics']
        artifact_paths = []

        for split in splits:
            for typ in types:
                plt.figure(figsize=(8, 6))
                found_any = False
                for label, result in results.items():
                    mean_key = f"{split}_{typ}_mean"
                    std_key = f"{split}_{typ}_std"
                    mean = result.get(mean_key)
                    std = result.get(std_key)
                    if mean is not None:
                        plt.plot(mean, label=f"{label} (mean)")
                        if std is not None:
                            plt.fill_between(
                                range(len(mean)),
                                mean - std,
                                mean + std,
                                alpha=0.2
                            )
                        found_any = True
                if not found_any:
                    plt.close()
                    continue

                plt.title(f"{split.capitalize()} {typ[:-2].capitalize()} (mean ± std over seeds)")
                plt.xlabel("Outer Step")
                plt.ylabel(typ[:-2].capitalize())
                plt.yscale(y_scale)  # Set the y-axis scale
                plt.legend()
                plt.tight_layout()
                path = os.path.join(output_dir, f"{split}_{typ}_mean_std.png")
                plt.savefig(path)
                plt.close()
                if _run:
                    _run.add_artifact(path)
                artifact_paths.append(path)

        return artifact_paths
