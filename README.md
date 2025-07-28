# Variance Reduction of Stochastic Hypergradient Estimation by Mixed Fixed-Point Iteration

This is the official implementation of the experiments in the following paper:

> Naoyuki Terashita and Satoshi Hara  
> [**Variance Reduction of Stochastic Hypergradient Estimation by Mixed Fixed-Point Iteration**](https://openreview.net/forum?id=mkmX2ICi5c)  
> *Transactions on Machine Learning Research*, 2025


## Setup
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Experiments

### Section 5.1: Effect of Mixing Rate

To reproduce the experiments in Section 5.1 (tuning alpha and scale parameters):

```bash
python main.py HypergradEstimationPipeline conf.paper_tune_alpha_and_scale
```

**Plots**: Results can be visualized using the notebook `notebooks/main_tune_alpha_and_scale.ipynb`

### Section 5.2: Comparison with Existing Approaches

To reproduce the benchmark experiments in Section 5.2:

```bash
# Fashion-MNIST dataset
python main.py HypergradEstimationPipeline conf.paper_fashion

# Adult dataset
python main.py HypergradEstimationPipeline conf.paper_adult

# California housing dataset
python main.py HypergradEstimationPipeline conf.paper_california

# Synthetic dataset
python main.py HypergradEstimationPipeline conf.paper_synth
```

**Plots**: Results can be visualized using the notebook `notebooks/main_benchmark.ipynb`

### Appendix: Hyperparameter Optimization

For additional bilevel optimization experiments:

```bash
python main.py BilevelOptimizationPipeline conf.app_adult
```

## Visualization

Results and plots are generated using Jupyter notebooks in the `notebooks/` directory:

- `main_tune_alpha_and_scale.ipynb` - Plots for Section 5.1
- `main_benchmark.ipynb` - Plots for Section 5.2
- `app_bo.ipynb` - Plots for Appendix

## Citation

If you use this code in your research, please cite our paper:
```
@article{
terashita2025variance,
title={Variance Reduction of Stochastic Hypergradient Estimation by Mixed Fixed-Point Iteration},
author={Naoyuki Terashita and Satoshi Hara},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=mkmX2ICi5c},
}
```



---
If you have questions, please contact Naoyuki
Terashita ([naoyuki.terashita.sk@hitachi.com](mailto:naoyuki.terashita.sk@hitachi.com)).
