from torcheeg.io.eeg_signal import EEGSignalIO
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Cached_Dataset(Dataset):
    def __init__(self, cache_path, idxs=None):
        self.io = EEGSignalIO(io_path=str(cache_path), io_mode='lmdb')
        self.metadata = pd.read_csv(cache_path / 'metadata.csv', index_col=0)
        self.metadata.drop('Unnamed: 0', axis=1, inplace=True) ## This should be fixed in the cache creation
        self.set_split(idxs)
        self.samples = 0
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        datafile = self.metadata.loc[idx, 'data_file'] # This is an added lookup.. we could remove it by not needing the split 
        return self.io.read_eeg(str(datafile))[:,:1024], self.y[idx]
    
    def set_split(self, idxs):
        self.metadata.reset_index(drop=True, inplace=True)
        self.metadata['data_file'] = self.metadata.index
        le = LabelEncoder()
        self.metadata['label_encoding'] = le.fit_transform(self.metadata['label'])
        
        if idxs is not None:
            self.metadata = self.metadata.loc[idxs]
            self.metadata.reset_index(drop=True, inplace=True)

        self.y = self.metadata['label_encoding'].values
        self.subjects = self.metadata['subject'].values