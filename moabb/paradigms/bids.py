import abc
import logging
from moabb.datasets import utils
from moabb.datasets.preprocessing import FindBIDSEvents
from moabb.paradigms.base import BaseParadigm

log = logging.getLogger(__name__)

class BIDSclassification(BaseParadigm):
    """Assumes that the dataset is in BIDS format and that the task is classification."""

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
        # This should be a check for BIDS format
        # we should verify list of channels, somehow (my comment: BIDS way)
        
        ret = True
        if not (dataset.paradigm == "BIDS"):
            ret = False

        return ret
    
    # def used_events(self, dataset):
    #     return {ev: dataset.event_id[ev] for ev in self.events}
    
    def _get_events_pipeline(self, dataset):
        # event_id = self.used_events(dataset)
        return FindBIDSEvents(event_id=event_id, interval=dataset.interval, layout=dataset.layout)

    @property
    def datasets(self):
        if self.tmax is None:
            interval = None
        else:
            interval = self.tmax - self.tmin
        return utils.dataset_search(
            paradigm="BIDS", events=self.events, interval=interval, has_all_events=True
        )

    @property
    def scoring(self):
        return "accuracy"