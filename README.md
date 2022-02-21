# Robust Geometric Metric Learning - RGML

![Build](https://github.com/antoinecollas/robust_metric_learning/workflows/tests/badge.svg)


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

...
