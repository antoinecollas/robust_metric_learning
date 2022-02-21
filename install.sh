# create conda environment
conda create -n robust_metric_learning python=3.7 --yes 
eval "$(conda shell.bash hook)"  # bug fix: https://github.com/conda/conda/issues/7980#issuecomment-492784093 
conda activate robust_metric_learning

# install libraries
pip install -r requirements.txt
pip install --upgrade "jax[cpu]"
pip install git+https://github.com/scikit-learn-contrib/metric-learn
pip install git+https://github.com/antoinecollas/pymanopt

# install package
python setup.py install
