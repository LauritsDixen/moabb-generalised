from pathlib import Path
from bids import BIDSLayout

from moabb.datasets.base import BaseDataset
import mne


class FACED(BaseDataset):
    """FACED dataset.
    Custom emotion recognition dataset.
    """

    def __init__(self, BIDS_path, subjects=None):
        print("Loading BIDS layout...")
        self.layout = BIDSLayout(BIDS_path)
        print("BIDS layout loaded.")
        eeg_files = self.layout.get(return_type='fir', datatype='eeg', suffix='eeg', extension='edf')
        valids = list(set([v.entities['subject'] for v in eeg_files]))
        if subjects is not None:
            valids = [v for v in valids if v in subjects]
        self.subject_map = {i:s for i,s in enumerate(valids)}

        super().__init__(
            subjects=list(self.subject_map.keys()),
            sessions_per_subject=1,
            events={emo:i for i, emo in enumerate(["anger", "disgust", "fear", "sadness", "neutral", "amusement", "inspiration", "joy", "tenderness"])},
            code="FACED",
            interval=[0, 4],
            paradigm="emotion",
            doi="",
        )
        
    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        print(f"Loading data for subject {self.subject_map[subject]}...")
        file_path_list = self.data_path(subject)

        sessions = {'0':{}}
        for edf_file in file_path_list:
            run = str(edf_file.entities['run'])
            print(f"Loading run {run}")
            raw = mne.io.read_raw_edf(edf_file.path, preload=True, verbose=False)
            sessions['0'][run] = raw

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Fetch the data from one subject."""
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))
        subject = self.subject_map[subject]
        eeg_files = self.layout.get(return_type='fir', datatype='eeg', suffix='eeg', extension='edf', subject=subject)
        return eeg_files