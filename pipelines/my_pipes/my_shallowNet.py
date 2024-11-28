import torch
from braindecode import EEGClassifier
from braindecode.models import ShallowFBCSPNet
from sklearn.pipeline import Pipeline
from skorch.callbacks import EarlyStopping, EpochScoring, WandbLogger
from skorch.dataset import ValidSplit

from moabb.utils import setup_seed
import wandb

from moabb.pipelines.features import Resampler_Epoch


local = True
device = "mps"

SEED = 42
setup_seed(SEED)

# Hyperparameter
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 2**7
VERBOSE = 0
EPOCH = 3000
PATIENCE = 100
DROPOUT = 0.5
if local:
    EPOCH = 2

callbacks = [
    EarlyStopping(monitor="valid_loss", patience=PATIENCE, load_best=True),
    EpochScoring(
        scoring="accuracy", on_train=True, name="train_acc", lower_is_better=False
    ),
    EpochScoring(
        scoring="accuracy", on_train=False, name="valid_acc", lower_is_better=False
    ),
    # EpochScoring(
    #     scoring="roc_auc", on_train=True, name="train_roc_auc", lower_is_better=False
    # ),
    # EpochScoring(
    #     scoring="roc_auc", on_train=False, name="valid_roc_auc", lower_is_better=False
    # ),
    EpochScoring(
        scoring="f1", on_train=True, name="train_f1", lower_is_better=False
    ),
    EpochScoring(
        scoring="f1", on_train=False, name="valid_f1", lower_is_better=False
    )
]

if wandb.run is not None:
    callbacks.append(WandbLogger(wandb.run))

# Define a Skorch classifier
clf = EEGClassifier(
    module=ShallowFBCSPNet,
    module__final_conv_length="auto",
    optimizer=torch.optim.Adam,
    optimizer__lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    max_epochs=EPOCH,
    train_split=ValidSplit(0.2, random_state=SEED, stratified=True),
    device=device,
    callbacks=callbacks,
    verbose=VERBOSE,  # Not printing the results for each epoch
)

# Create the pipelines
pipes = Pipeline(
    [
        ("resample", Resampler_Epoch(250)),
        ("ShallowFBCSPNet", clf),
    ]
)

# this is what will be loaded
PIPELINE = {
    "name": "my_ShallowFBCSPNet",
    "paradigms": ["LeftRightImagery", "MotorImagery"],
    "return_epochs": True,
    "pipeline": pipes,
}
