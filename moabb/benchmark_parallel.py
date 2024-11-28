import logging
import importlib
from pathlib import Path

# import mne
# import pandas as pd
import yaml
import wandb

from moabb import paradigms as moabb_paradigms
from moabb.evaluations import (
    MixedSubjectSingleEvaluation,
    CrossSubjectSingleEvaluation
)
from moabb.analysis import analyze

from moabb.pipelines.utils import (
    # generate_paradigms,
    # parse_pipelines_from_directory,
    create_pipeline_from_config,
)


try:
    from codecarbon import EmissionsTracker  # noqa

    _carbonfootprint = True
except ImportError:
    _carbonfootprint = False

log = logging.getLogger(__name__)


class BenchmarkParallel():

    # noqa: C901
    def init_sweep(
            self,  
            evaluations,
            datasets,
            paradigms,
            pipelines_folder,
            results_folder="./results/",
            save_model=False,
            overwrite=False,
            name="test",
            random_state=42,
            n_splits=5,
            max_folds=None,
            cache_config=None,
            project=None,
        ):

        pipelines_folder = Path(pipelines_folder)
        # count number of pipelines
        pipelines = [f.name for f in pipelines_folder.glob("*.py")] \
                  + [f.name for f in pipelines_folder.glob("*.yml")]

        if isinstance(datasets, str):
            datasets = [datasets]
        if isinstance(paradigms, str):
            paradigms = [paradigms]
        if isinstance(evaluations, str):
            evaluations = [evaluations]

        runs = []
        for paradigm in paradigms:
            p = getattr(moabb_paradigms, paradigm)() # instatiated paradigm
            pds = [d for d in p.datasets if d.code in datasets]
            for dataset in pds:
                n_subjects = len(dataset.subject_list)
                if max_folds is not None:
                    n_subjects = min(n_subjects, max_folds)
                    n_splits = min(n_splits, max_folds)                    
                
                # assumed that all datasets are compatible with all pipelines
                # Possible to check with yaml reading instead of py of braindecode pipes
                for pipeline in pipelines: 
                    for evaluation in evaluations:
                        # Defensive programming
                        evaluation = evaluation.replace("Evaluation", "").replace("Single", "")
                        run_config = {
                            "dataset": dataset.code,
                            "paradigm": paradigm,
                            "pipeline": pipeline,
                            "evaluation": evaluation
                        }
                        if evaluation == "CrossSubject":
                            for subject_idx in range(n_subjects):
                                fold_config = run_config.copy()
                                fold_config["fold"] = subject_idx
                                runs.append(fold_config)
                        elif evaluation == "MixedSubject":
                            for fold in range(n_splits):
                                fold_config = run_config.copy()
                                fold_config["fold"] = fold
                                runs.append(fold_config)
                        else:
                            raise ValueError(f"Unknown evaluation: {evaluation}")

        results_folder = str(Path(results_folder) / name)
        sweep_configuration = {
            "method": "grid",
            "name": name,
            "metric": {"goal": "minimize", "name": "loss"},
            "parameters": {
                "runs": {"values": runs},
                ## Parameters to keep constant
                "save_model": {"value": save_model}, # bool
                "overwrite": {"value": overwrite}, # bool
                "results_folder": {"value": results_folder}, # str
                "pipelines_folder" : {"value": str(pipelines_folder)}, # str
                "n_splits": {"value": n_splits}, # int
                "cache_config": {"value": cache_config}, # dict of bools
                "random_state": {"value": random_state}, # int
            }
        }

        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)
        print(f"Sweep created with {len(runs)} runs")
        return sweep_id

    def _run_experiment(self):
        wandb_run = wandb.init()
        configs = wandb_run.config
        run_configs = configs['runs']

        valid_evaluations = {
            "MixedSubject": MixedSubjectSingleEvaluation,
            "CrossSubject": CrossSubjectSingleEvaluation,
        }
        pdrgm_name = run_configs['paradigm']
        paradigm = getattr(moabb_paradigms, pdrgm_name)()
        
        eval_name = run_configs['evaluation']
        evaluation = valid_evaluations[eval_name]

        dataset_name = run_configs['dataset']
        dataset = [d for d in paradigm.datasets if d.code == dataset_name]

        if len(dataset) == 0:
            print("Dataset not found")
            wandb.finish()
            return None

        # Load pipeline
        pipefile = Path(configs['pipelines_folder']) / run_configs['pipeline']
        pipe_configs = self.read_config(pipefile) # instantiates pipeline
        pipeline = {
            pipe_configs['name'] : pipe_configs['pipeline']
        }

        return_epochs = pipe_configs.get('return_epochs', False)
        suffix = f"ds-{dataset_name}_pipe-{run_configs['pipeline']}_fold-{run_configs['fold']}"
        res = evaluation(
            paradigm=paradigm,
            datasets=dataset,
            return_epochs=return_epochs,
            n_splits=configs['n_splits'],
            random_state=configs['random_state'],
            overwrite=configs['overwrite'],
            save_model=configs['save_model'],
            hdf5_path=configs['results_folder'],
            suffix=suffix,
            cache_config=configs['cache_config'],
            fold=run_configs['fold']
        ).process(
            pipeline=pipefile
        )

        print(f"Results: {res}")
        out_folder = Path(configs['results_folder'])
        print(f"Saving results to {out_folder}")
        if not out_folder.exists():
            out_folder.mkdir(parents=True)
        analyze(res, str(out_folder), name=wandb_run.name, plot=False)
        wandb_run.finish()
    
    def run_sweep(self, sweep_id, entity=None, project=None, count=None):
        wandb.agent(sweep_id, function=self._run_experiment, entity=entity, project=project, count=count)
        print("Run completed")

    def read_config(self, config_file):
        if config_file.suffix == ".yml":
            with open(config_file, "r") as _file:
                content = _file.read()
                config_dict = yaml.load(content, Loader=yaml.FullLoader)
                ppl = create_pipeline_from_config(config_dict["pipeline"])
                config_dict['pipeline'] = ppl
            return config_dict
        if config_file.suffix == ".py":
            spec = importlib.util.spec_from_file_location("custom", config_file)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
            return foo.PIPELINE
        else:
            raise ValueError("Invalid file format. Supported formats are .yml and .py")
