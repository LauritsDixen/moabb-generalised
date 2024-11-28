import abc
import logging

import pandas as pd
from tqdm import tqdm

from moabb.datasets import utils
from moabb.datasets.preprocessing import FindBIDSEvents
from moabb.paradigms.base import BaseParadigm




log = logging.getLogger(__name__)


class BaseEmotionRecognition(BaseParadigm):
    """Base Emotion recognition paradigm."""

    def __init__(
        self,
        filters=([7, 35],),
        events=None,
        tmin=0.0,
        tmax=None,
        baseline=None,
        channels=None,
        resample=None,
    ):
        super().__init__(
            filters=filters,
            events=events,
            channels=channels,
            baseline=baseline,
            resample=resample,
            tmin=tmin,
            tmax=tmax,
        )

    def is_valid(self, dataset):
        ret = True
        if not (dataset.paradigm == "emotion"):
            ret = False

        # check if dataset has required events
        if self.events:
            if not set(self.events) <= set(dataset.event_id.keys()):
                ret = False

        # we should verify list of channels, somehow
        return ret

    @abc.abstractmethod
    def used_events(self, dataset):
        pass

    @property
    def datasets(self):
        if self.tmax is None:
            interval = None
        else:
            interval = self.tmax - self.tmin
        return utils.dataset_search(
            paradigm="emotion", events=self.events, interval=interval, has_all_events=True
        )

    @property
    def scoring(self):
        return "accuracy"
    
    
class SinglePassEmotion(BaseEmotionRecognition):
    """Single Bandpass filter emotion recognition."""

    def __init__(self, fmin=8, fmax=32, **kwargs):
        if "filters" in kwargs.keys():
            raise (ValueError("SinglePassParadigms do not take argument filters"))
        super().__init__(filters=[[fmin, fmax]], **kwargs)

class DiscreteEmotions(SinglePassEmotion):
    """Emotion recognition for discrete emotions."""

    def __init__(self, **kwargs):
        if "events" in kwargs.keys():
            raise (ValueError("DiscreteEmotions dont accept events"))
        super().__init__(events=["sad", "happy", "neutral", "fear", "angry", "disgust"], **kwargs)

    def used_events(self, dataset):
        return {ev: dataset.event_id[ev] for ev in self.events}
    
    def _get_events_pipeline(self, dataset):
        event_id = self.used_events(dataset)
        return FindBIDSEvents(event_id=event_id, interval=dataset.interval, layout=dataset.layout)
