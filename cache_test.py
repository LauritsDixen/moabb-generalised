from moabb import paradigms as moabb_paradigms
from moabb.evaluations import (
    MixedSubjectSingleEvaluation,
    CrossSubjectSingleEvaluation,
)
from moabb.datasets import FER


data_path = '/Users/ldix/Documents/Projects/fer_BIDS'
evaluation = CrossSubjectSingleEvaluation
dataset = FER(data_path)
paradigm = moabb_paradigms.DiscreteEmotions()
pipefile = "/Users/ldix/Documents/Projects/moabb-generalised/pipelines/braindecode_shallowFSCBPNet.yml"

res = evaluation(
    paradigm=paradigm,
    datasets=dataset,
    # n_splits=configs['n_splits'],
    # random_state=configs['random_state'],
    # overwrite=configs['overwrite'],
    # save_model=configs['save_model'],
    # hdf5_path=configs['results_folder'],
    # suffix=suffix,
    # cache_config=configs['cache_config'],
    fold=0#run_configs['fold']
 ).process(
    pipefile=pipefile
)

print(f"Results: {res}")