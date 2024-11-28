import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# import logging

# from argparse import ArgumentParser
# import mne

# from moabb.benchmark import benchmark
from moabb.benchmark_parallel import BenchmarkParallel

from pathlib import Path

cwd = Path.cwd()


start_sweep = 0

if __name__ == "__main__":
    bp = BenchmarkParallel()
    if start_sweep:
        sweep_id = bp.init_sweep(
            evaluations=["MixedSubject"],
            paradigms=["LeftRightImagery"],
            datasets=['BNCI2014-004'],
            pipelines_folder="./pipelines/my_pipes/",
            results_folder="./sweep_results/",
            save_model=False,
            overwrite=True,
            name="test",
            n_splits=5,
            max_splits=None,
            random_state=42,
            cache_config=None,
            project="moabb-testing",
    )
    else:
        bp.run_sweep("7fadas69", entity=None, project="moabb-testing", count=1)