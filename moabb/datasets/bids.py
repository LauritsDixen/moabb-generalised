from pathlib import Path
from bids import BIDSLayout

from moabb.datasets.base import BaseDataset
import mne


class BIDS(BaseDataset):
    """BIDS-based dataset. Should be compatible with any BIDS dataset. 
    """

    def __init__(self, BIDS_path, subjects=None, sessions=None, runs=None, task=None, eeg_ext='edf'):
        print("Loading BIDS layout...")
        self.layout = BIDSLayout(BIDS_path)
        print("BIDS layout loaded.")

        self.runs = self.layout.get_runs() if runs is None else runs
        self.sessions = self.layout.get_sessions() if sessions is None else sessions
        self.subjects = self.layout.get_subjects() if subjects is None else subjects
        self.tasks = self.layout.get_tasks() if task is None else task
        self.eeg_ext = eeg_ext

        eeg_files = self.layout.get(suffix='eeg',
                                    run=self.runs, 
                                    task=self.tasks,
                                    subject=self.subjects,
                                    session=self.sessions, 
                                    extension=self.eeg_ext)
        valid_subjects = list(set([v.entities['subject'] for v in eeg_files]))
        # subject map is needed, since the subject can be named anything, not just numbers
        self.subject_map = {i:s for i,s in enumerate(valid_subjects)}
        self.subjects = list(self.subject_map.keys())
        self.n_sessions = max(len(self.sessions), 1)

        super().__init__(
            subjects=self.subjects,
            sessions_per_subject=self.n_sessions,
            code="BIDS",
            paradigm="BIDS",
            # we don't know the interval and events yet, these are found by paradigms
            interval=None, 
            events={}
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""

        assert subject in self.subject_list, "Invalid subject number"

        subject = self.subject_map[subject]
        eeg_files = self.layout.get(suffix='eeg',
                                    task=self.tasks,
                                    subject=subject,
                                    session=self.sessions, 
                                    run=self.runs, 
                                    extension=self.eeg_ext
                                    )

        sessions = {}
        for egg in eeg_files:
            raw = self.read_raw(egg)
            session = egg.entities.get('session', 1)
            run = egg.entities.get('run', 1)
            sessions[session] = sessions.get(session, {})
            sessions[session][run] = raw

        return sessions
    
    def read_raw(self, egg):
        """
        Add information to the raw object, so far only in the temp field, but everything is stored there.
        So far only supports edf files.
        """

        if self.eeg_ext == 'edf':
            raw = mne.io.read_raw_edf(egg, preload=True, verbose=False, include='eeg')
        else:
            raise ValueError(f"Extension {self.eeg_ext} not supported.")
        ## this can be useful, if data is just raw data streams, needs to be fixed
            channels_file = self.layout.get_nearest(path = egg.path.replace('eeg.edf','channels.tsv'), suffix='channels', return_type='id')
            channels = pd.read_csv(channels_file.path, sep='\t')
            info = mne.create_info(ch_names=channels['name'].values.tolist(), sfreq=raw.info['sfreq'], ch_types=channels['type'].values.tolist())
            raw = mne.io.read_raw_edf(datafile, preload=True, verbose=False, info=info)
        raw.info['temp'] = egg.get_metadata()
        raw.info['temp']['bids_file'] = egg

        return raw
    
    def data_path(self):
        return self.layout.root
    
