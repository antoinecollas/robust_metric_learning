# Robust Geometric Metric Learning - RGML

![Build](https://github.com/antoinecollas/robust_metric_learning/workflows/tests/badge.svg)

This repository hosts Python code for Robust Geometric Metric Learning.

See the associated [arXiv paper](https://arxiv.org/abs/2202.11550).


## Installation

The script `install.sh` creates a conda environment with everything needed to run the examples of this repo and installs the package:

```
./install.sh
```

## Check

To check the installation, activate the created conda environment `robust_metric_learning` and run the unit tests:

```
conda activate robust_metric_learning
nose2 -v --with-coverage
```


## Run experiments

To run experiments, run the scripts of `examples/`, e.g.

```
python examples/mislabeling.py
```


## Cite

If you use this code please cite:

```
@misc{collas2022robust,
      title={Robust Geometric Metric Learning}, 
      author={Antoine Collas and Arnaud Breloy and Guillaume Ginolhac and Chengfang Ren and Jean-Philippe Ovarlez},
      year={2022},
      eprint={2202.11550},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
