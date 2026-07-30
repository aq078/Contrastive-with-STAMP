from stamp.experiments import (
    run_full_experiment
)
from stamp.local import get_local_config
import os

local_config = get_local_config()

dataset_names = ['Penn_Action']

#change before run
exp_names = [
'Penn_Action_stage2'
]

experiments_dir = local_config.tsfm_experiments_dir
device = 'cuda:0'

for dataset_name in dataset_names:
    for exp_name in exp_names:

        if os.path.exists(os.path.join(experiments_dir, dataset_name, exp_name, 'figures')):
            print(f"Experiment {exp_name} for dataset {dataset_name} already has a /figures folder. Skipping...")
            continue
        run_full_experiment(dataset_name, exp_name, experiments_dir, device, exp_type='embedding')