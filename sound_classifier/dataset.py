import numpy as np
from torch.utils.data import Dataset

class SpecDataset(Dataset):
    def __init__(self, paths, mean=0, std=1, time_dim_size=None):
        self.paths = paths
        self.mean = mean
        self.std = std
        self.time_dim_size = time_dim_size

    def __getitem__(self, i):
        # Get i-th path.
        path = self.paths[i]
        # Get i-th spectrogram path.
        path = 'gtzan/spec/' + path.replace('.wav', '.npy')

        # Extract the genre from its path.
        genre = path.split('/')[-2]
        # Trun the genre into index number.
        label = genre_dict[genre]

        # Load the mel-spectrogram.
        spec = np.load(path)
        if self.time_dim_size is not None:
            # Slice the temporal dimension with a fixed length so that they have
            # the same temporal dimensionality in mini-batches.
            spec = spec[:, :self.time_dim_size]
        # Perform standard normalization using pre-computed mean and std.
        spec = (spec - self.mean) / self.std

        return spec, label

    def __len__(self):
        return len(self.paths)