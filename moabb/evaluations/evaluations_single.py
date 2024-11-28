from moabb.evaluations.base import BaseEvaluation
from moabb.evaluations.utils import (
    create_save_path,
    save_model_cv,
    create_EEGClassifier_pipeline_from_config
)
from moabb.datasets import Cached_Dataset

from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
)
from sklearn.model_selection._validation import _score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import get_scorer

from mne.epochs import BaseEpochs

import pandas as pd
import numpy as np
from time import time
from pathlib import Path


class SingleEvaluation(BaseEvaluation):

    def __init__(self, fold, **kwargs):
        self.fold = fold
        n_splits = 5 if 'n_splits' not in kwargs.keys() else kwargs['n_splits']
        super().__init__(n_splits=n_splits, **kwargs)

    def is_valid(self, dataset):
        return len(dataset.subject_list) > 0

    def process(self, pipefile, postprocess_pipeline=None):
        """
        Run the Braindecode pipeline on the dataset. Stands in place of the .process() method of the BaseEvaluation class.
        This method calls the evaluate() method and handles results.
        The main difference is we read the model_configs from yml file and instantiate the clf here.
        This is to make sure the model can take metadata from the dataset and we do not load the data upfront. 
        """

        ## Load the pipeline from the yml file
        if not isinstance(pipefile, Path):
            if isinstance(pipefile, str):
                pipefile = Path(pipefile)
            else:
                raise ValueError("pipefile must be a Path or a str")

        dataset = self.datasets[0] if isinstance(self.datasets, list) else self.datasets
        print(f"Processing dataset for caching: {dataset.code}")
        process_pipeline = self.paradigm.make_process_pipelines(
            dataset,
            return_epochs=self.return_epochs,
            return_raws=self.return_raws,
            postprocess_pipeline=postprocess_pipeline,
        )[0]
        # (we only keep the pipeline for the first frequency band, better ideas?)
        # TODO: we need to reimplment this part anyway.
        
        pipeline = create_EEGClassifier_pipeline_from_config(pipefile, n_classes=len(dataset.event_id))

        # We expect a single dataset
        # Dataset is of type: moabb.datasets.base.BaseDataset
        self.dataset = self.datasets[0] if isinstance(self.datasets, list) else self.datasets
    
        # get the datapath
        cache_path = self.paradigm.run_cache(
            dataset=dataset,
            subjects=None,
            postprocess_pipeline=postprocess_pipeline,
        )

        results = self.evaluate(
            cache_path=cache_path,
            pipeline=pipeline,
            process_pipeline=process_pipeline,
            postprocess_pipeline=postprocess_pipeline,
        )

        results['dataset'] = dataset

        self.push_result(results, pipeline, process_pipeline)
        return self.results.to_dataframe(
            pipelines=pipeline, process_pipeline=process_pipeline
        )
    

class MixedSubjectSingleEvaluation(SingleEvaluation):

    # flake8: noqa: C901
    def evaluate(
        self, cache_path, pipeline, process_pipeline=None, postprocess_pipeline=None
    ):
        pname = cache_path.parent.name
        dcode = cache_path.name

        pipename, clf = list(pipeline.items())[0]
        print(f"Start training with: {dcode}, {pname}, {pipename}, {self.__class__.__name__}")
        
        ds = Cached_Dataset(cache_path)
        fold_idx = self.fold

        train_idx, test_idx = [], []
        for subject in np.unique(ds.subjects):
            sss = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            subj_metadata = ds.metadata[ds.metadata['subject'] == subject]
            y_subj = subj_metadata['subject'].values
            train_idx_subj, test_idx_subj = list(sss.split(subj_metadata, y_subj))[fold_idx]
            train_idx.append(subj_metadata.index[train_idx_subj])
            test_idx.append(subj_metadata.index[test_idx_subj])

        train = np.concatenate(train_idx)
        test = np.concatenate(test_idx)
        train_data = Cached_Dataset(cache_path, train)
        test_data = Cached_Dataset(cache_path, test)

        scorer = get_scorer(self.paradigm.scoring)
        t_start = time()
        model = clf.fit(train_data, train_data.y)
        duration = time() - t_start
        print(f"Training done in {duration} seconds")

        if self.hdf5_path is not None and self.save_model:
            print("Saving model")
            model_save_path = create_save_path(
                hdf5_path=self.hdf5_path,
                code=dcode,
                subject="",
                session="",
                name=pipename,
                grid=False,
                eval_type="MixedSubject",
            )
            save_model_cv(
                model=model, save_path=model_save_path, cv_index=str(fold_idx+1)
            )

        print("Scoring model")
        score = _score(
            estimator=model,
            X_test=test_data,
            y_test=test_data.y,
            scorer=scorer,
            score_params={},
        )
        nchan = train_data[0][0].shape[0]
        res = {
            "time": duration,
            "dataset": dcode,
            "subject": "mixed",
            "session": "mixed",
            "score": score,
            "n_samples": len(train),
            "n_channels":nchan,
            "pipeline": pipename,
        }
        print(f"Score: {score}")

        return res

class CrossSubjectSingleEvaluation(SingleEvaluation):

    def is_valid(self, dataset):
        return len(dataset.subject_list) > 1

    # flake8: noqa: C901
    def evaluate(
        self, cache_path, pipeline, process_pipeline=None, postprocess_pipeline=None
    ):
        pname = cache_path.parent.name
        dcode = cache_path.name

        pipename, clf = list(pipeline.items())[0]
        print(f"Processing dataset: {dcode}, paradigm: {pname}, pipeline: {pipename}")
        
        ds = Cached_Dataset(cache_path)
        subject_idx = self.fold # fold is the test subject index

        cv = LeaveOneGroupOut()
        idxs = np.arange(len(ds))
        train, test = list(cv.split(idxs, groups=ds.subjects))[subject_idx]
        train_data = Cached_Dataset(cache_path, train)
        test_data = Cached_Dataset(cache_path, test)

        scorer = get_scorer(self.paradigm.scoring)
        t_start = time()
        model = clf.fit(train_data, train_data.y)
        duration = time() - t_start

        if self.hdf5_path is not None and self.save_model:
            model_save_path = create_save_path(
                hdf5_path=self.hdf5_path,
                code=dcode,
                subject=subject_idx,
                session="",
                name=pipename,
                grid=False,
                eval_type="CrossSubject",
            )
            save_model_cv(
                model=model, save_path=model_save_path, cv_index=str(subject_idx)
            )
        
        nchan = train_data[0][0].shape[0]
        score = _score(
            estimator=model,
            X_test=test_data,
            y_test=test_data.y,
            scorer=scorer,
            score_params={},
        )
        res = {
            "time": duration,
            "dataset": dcode,
            "subject": subject_idx,
            "session": "mixed",
            "score": score,
            "n_samples": len(train),
            "n_channels": nchan,
            "pipeline": pipename,
        }
        return res