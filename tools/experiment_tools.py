import os
import numpy as np

def perform(experiment, condition, experiment_configs, data_path, save_path):
    """Train all models locally."""
    for i in range(0, 1000):
        config = vary_config(experiment_configs, i)
        if config:
            print('[***] Hyper-parameter: %2d' % i)
            current_save_path = os.path.join(save_path, str(i).zfill(6))
            experiment(condition, config, data_path, current_save_path)


def vary_config(experiment, i):
    """Training a specific hyperparameter settings.
    Args:
        experiment: a tuple (config, hp_ranges)
        i: integer, indexing the specific hyperparameter setting to be used
       hp['a']=[0,1], hp['b']=[0,1], hp['c']=[0,1], there are 8 possible combinations
    Return:
        config: new configuration
    """
    # Ranges of hyperparameters to loop over
    config, hp_ranges = experiment

    # Unravel the input index
    keys = hp_ranges.keys()
    dims = [len(hp_ranges[k]) for k in keys]
    n_max = np.prod(dims)
    indices = np.unravel_index(i % n_max, dims=dims)

    if i >= n_max:
        return False

    # Set up new hyperparameter
    for key, index in zip(keys, indices):
        setattr(config, key, hp_ranges[key][index])
    return config

