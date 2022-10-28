import pickle

import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical


class CovidCoughDataset(Sequence):
    def __init__(self, saved_data, batch_size):
        with saved_data.open("rb") as data:
            processed_audio = pickle.load(data)
        self.data = np.array(
            [processed_audio[fname]["audio"] for fname in sorted(processed_audio)]
        )
        self.labels = np.array(
            [processed_audio[fname]["covid19"] for fname in sorted(processed_audio)]
        )
        self.labels = to_categorical(self.labels)
        self.data = np.expand_dims(self.data, -1)
        self.batch_size = batch_size
        self.indices = np.arange(len(self.data))
        np.random.shuffle(self.indices)

    def set_fold(self, fold_index, total_folds):
        n = len(self.data) // total_folds
        self.indices = np.arange(fold_index * n, (fold_index + 1) * n)
        np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices) // self.batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        data = self.data[
            self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        ]
        label = self.labels[
            self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        ]
        return data, label
