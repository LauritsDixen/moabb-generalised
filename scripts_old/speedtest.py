import pandas as pd
import numpy as np
import pickle as pkl
from pathlib import Path
from time import time
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torcheeg.io.eeg_signal import EEGSignalIO
from mne import EpochsArray
import mne

class MNEDataset(Dataset):
    def __init__(self, metadata):
        self.metadata = metadata
        self.name = 'mne'

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        filepath =  self.metadata.loc[idx, 'filepath']
        y = self.metadata.loc[idx, 'label']
        x = EpochsArray.load(filepath)
        return x, y
    
    def save_file(self, x, metadata):
        filepath = metadata['filepath']
        info = mne.create_info(ch_names=[f'ch{i}' for i in range(x.shape[0])], sfreq=128, ch_types='eeg')
        x = EpochsArray(x, info)
        x.save(filepath)

class PickleDataset(Dataset):
    def __init__(self, metadata):
        self.metadata = metadata
        self.name = 'pickle'

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        filepath =  self.metadata.loc[idx, 'filepath']
        y = self.metadata.loc[idx, 'label']
        with open(filepath, 'rb') as f:
            x = pkl.load(f)
        return x, y
    
    def save_file(self, x, metadata):
        filepath = metadata['filepath']
        with open(filepath, 'wb') as f:
            pkl.dump(x, f)



class torchEEGDataset(Dataset):
    def __init__(self, metadata):
        self.metadata = metadata
        self.name = 'torchEEG'
        self.io = EEGSignalIO(io_path='./speed_test', io_mode='lmdb')

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        y = self.metadata.loc[idx, 'label']
        x = self.io.read_eeg(str(idx))
        return x, y
    
    def save_file(self, x, metadata):
        # metadata = metadata.to_dict()
        self.io.write_eeg(x)

class SpeedTester:

    def __init__(self, n_subjects, n_trials, n_channels, trial_length, sfreq, dataset):
        self.n_subjects = n_subjects
        self.n_trials = n_trials
        self.n_channels = n_channels
        self.trial_length = trial_length
        self.sfreq = sfreq
        self.num_files = n_subjects * n_trials
        self.file_size = n_channels * int(trial_length * sfreq)
        self.results_folder = Path('./speed_results')
        self.results_folder.mkdir(exist_ok=True)
        self.root_folder = Path('./speed_test')
        self.root_folder.mkdir(exist_ok=True)
        self.dataset = dataset
        print(f"\nFile_size = {self.file_size}")

    def make_metadata(self):
        metadata = []
        for i in range(self.n_subjects):
            for j in range(self.n_trials):
                filepath = self.make_file_path(i, j)
                metadata.append({'subject': i, 'trial': j, 'label': np.random.randint(1, 4), 'filepath': filepath})
        self.metadata = pd.DataFrame(metadata)

    def make_file_path(self, subj, trial):
        return self.root_folder / f'sub-{subj}_{trial}.pkl'

    def make_data(self):
        # Clear the folder
        for file_path in self.root_folder.iterdir():
            file_path.unlink()
        for i in tqdm(range(len(self.metadata)), desc='Making files'):
            metadata = self.metadata.iloc[i]
            x = np.random.randn(self.n_channels, int(self.trial_length * self.sfreq))
            self.dataset.save_file(x, metadata)

    def read_data(self):
        begin = time()
        for batch_ndx, x in enumerate(self.loader):
            pass
        end = time()
        for file_path in self.root_folder.iterdir():
            file_path.unlink()
        return end - begin

    def run(self):
        self.make_metadata()
        self.dataset = self.dataset(self.metadata)
        print(f"Dataset: {self.dataset.name}, {len(self.dataset)} files")
        self.loader = DataLoader(self.dataset, batch_size=128, shuffle=True, num_workers=0)
        self.make_data()
        time_taken = self.read_data()

        results = {
            'time_taken': time_taken,
            'num_files': self.num_files,
            'file_size': self.file_size,
            'n_subjects': self.n_subjects,
            'n_trials': self.n_trials,
            'n_channels': self.n_channels,
            'trial_length': self.trial_length,
            'sfreq': self.sfreq,
        }

        return results

        

def debug_run(dataset):
    st = SpeedTester(10, 100, 10, 0.3, 64, dataset)
    print(f"Time taken: {st.run()['time_taken']:.2f} seconds")


def run_test(method, handler, settings):
    idx = 0
    results_file = Path('./speed_results') / f'results_{method}.csv'
    df = pd.DataFrame(columns=['time_taken', 'num_files', 'file_size', 'n_subjects', 'n_trials', 'n_channels', 'trial_length', 'sfreq'])
    for n_subjects in settings['n_subjects']:
        for n_trials in settings['n_trials']:
            for n_channels in settings['n_channels']:
                for trial_length in settings['trial_length']:
                    for sfreq in settings['sfreq']:
                        st = SpeedTester(n_subjects, n_trials, n_channels, trial_length, sfreq, handler)
                        results = st.run()
                        idx += 1
                        print(f"Run {idx}: {results['time_taken']:.2f} seconds")
                        df.loc[idx] = results
                        df.to_csv(results_file, index=False)




def analyse_results():
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()

    results_folder = Path('./speed_results')
    for file_path in results_folder.iterdir():
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            try:
                times = df['time_taken']
            except KeyError:
                times = df['time']
            sizes = df['file_size']
            ax.scatter(sizes, times, label=file_path.name.replace('.csv', '').replace('results_', ''))
    ax.set_xlabel('File size')
    ax.set_ylabel('Time taken')
    ax.legend()

    plt.show()

settings = {
    'n_subjects' : [10],
    'n_trials' : [1000],
    'n_channels' : [14, 64],
    'trial_length' : [0.3, 5],
    'sfreq' : [128, 512]
}

method = 'mne'
dataset = MNEDataset #torchEEGDataset, PickleDataset, MNEDataset




if __name__ == '__main__':
    debug_run(dataset)
    # run_test(method, dataset, settings)
    # analyse_results()