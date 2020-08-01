# Bayes-Adaptive Deep Model-Based Policy Optimisation

## Prerequisites

Before installing the RoMBRL package, make sure the following Python packages are installed:
 * [rllab](https://github.com/rll/rllab)
 * [MuJoCo 1.31](https://www.roboti.us/license.html)

Other dependencies can be installed with `pip install -r requirements.txt`.

## Running Experiments

Experiments for a particular environment can be run using:

```
python ALGORITHM
    --params     PARAMS  (required) Path to parameters file.
    --outputdir  OUTPUT  (required) Output directory.
    --logdir     LOGDIR  (optional) Path to log file.
    --policy	 POLICY  (optional) Policy type [bnn, lstm], default: bnn.
```

where `ALGORITHM` can be either `online_bnn_trpo.py` or `bnn_trpo.py`.

The default parameters that can be used to reproduce our results are stored under directory `params/default/`.

Trial data, dynamic functional samples as well as policy network will be saved each iteration under `OUTPUT` directory.