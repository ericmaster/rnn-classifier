import os
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

class EnviromentalDataset(Dataset):
    '''
    Dataset para el dataset de audio ambiental.
    '''
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, max_len):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.max_len = max_len

    def _get_audio_sample_path(self, index):
        fname = self.annotations.loc[index, 'filename']
        path = os.path.join(self.audio_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No existe: {path}")
        return path

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            signal = T.Resample(sr, self.target_sample_rate)(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
        return signal

    def _cut_if_necessary(self, signal):
        max_len = self.max_len * self.target_sample_rate
        if signal.shape[1] > max_len:
            signal = signal[:, :max_len]
        return signal

    def _right_pad_if_necessary(self, signal):
        max_len = self.max_len * self.target_sample_rate
        if signal.shape[1] < max_len:
            pad = max_len - signal.shape[1]
            signal = F.pad(signal, (0, pad))
        return signal

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        signal, sr = torchaudio.load(audio_sample_path)
        label = int(self.annotations.loc[index, 'target'])

        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)

        mel = self.transformation(signal).to(torch.float32)  # [1, 64, T]
        return mel, label


class EnviromentalDataModule(pl.LightningDataModule):
    '''
    DataModule para el dataset de audio ambiental.
    '''
    def __init__(self, csv_file, root_dir, mel_transf, target_sr, max_len_s, batch_size, num_workers):
        super().__init__()
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.mel_transf = mel_transf
        self.target_sr = target_sr
        self.max_len_s = max_len_s
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        full = EnviromentalDataset(self.csv_file, self.root_dir, self.mel_transf, self.target_sr, self.max_len_s)
        n = len(full)
        n_train = int(0.8 * n)
        n_val = (n - n_train) // 2
        n_test = n - n_train - n_val
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(full, [n_train, n_val, n_test])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


