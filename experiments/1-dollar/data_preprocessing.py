from torch.utils.data import DataLoader, random_split
import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import matplotlib.pyplot as plt


class UnistrokeDataset(Dataset):
    def __init__(self, human_dir, synth_dir, target_class, transform=None, max_length=70):
        self.human_files = self._get_files(human_dir, target_class)
        self.synth_files = self._get_files(synth_dir, target_class)
        self.transform = transform
        self.max_length = max_length

        self.data = []
        self.labels = []

        self._load_data()

    def _get_files(self, directory, target_class):
        pattern = os.path.join(directory, f"*{target_class}*.csv")
        matched_files = glob.glob(pattern)
        return matched_files

    def _load_data(self):
        # Load human-generated data (Positive Class: 1)
        for file in self.human_files:
            sequence = self._load_sequence(file)
            if sequence is not None:
                self.data.append(sequence)
                self.labels.append(1)

        # Load synthetic data (Negative Class: 0)
        for file in self.synth_files:
            sequence = self._load_sequence(file)
            if sequence is not None:
                self.data.append(sequence)
                self.labels.append(0)

    def _load_sequence(self, file_path):
        try:
            df = pd.read_csv(file_path, sep=' ')
            sequence = df[['x', 'y', 'time']].values

            # x -> dx/dt, y -> dy/dt
            sequence = self._compute_deltas(sequence)
            # Normalize
            sequence = (sequence - sequence.mean()) / sequence.std()
            sequence = self._pad_or_truncate(sequence)

            return sequence
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def _compute_deltas(self, sequence):
        deltas = np.diff(sequence, axis=0)

        delta_x = deltas[:, 0]
        delta_y = deltas[:, 1]
        delta_time = deltas[:, 2]

        epsilon = 1e-8
        delta_time = np.where(delta_time == 0, epsilon, delta_time)

        dx_dt = delta_x / delta_time  
        dy_dt = delta_y / delta_time  

        normalized_deltas = np.stack(
            (dx_dt, dy_dt), axis=1)

        return normalized_deltas

    def _pad_or_truncate(self, sequence):
        seq_len, input_size = sequence.shape
        if seq_len < self.max_length:
            padding = np.zeros((self.max_length - seq_len, input_size))
            sequence = np.vstack((sequence, padding))
        elif seq_len > self.max_length:
            sequence = sequence[:self.max_length, :]
        return sequence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'sequence': torch.tensor(self.data[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


def prepare_dataloaders_per_class(human_dir, synth_dir, target_class, batch_size=256, max_length=70, test_size=0.2, val_size=0.1):
    full_dataset = UnistrokeDataset(
        human_dir=human_dir,
        synth_dir=synth_dir,
        target_class=target_class,
        max_length=max_length
    )

    if len(full_dataset) == 0:
        raise ValueError(f"No data found for class '{target_class}'.")

    test_len = int(len(full_dataset) * test_size)
    if val_size != 0:
        train_val_len = len(full_dataset) - test_len
        val_len = int(train_val_len * val_size)
        train_len = train_val_len - val_len


        train_val_dataset, test_dataset = random_split(full_dataset, [
                                                    train_len + val_len, test_len], generator=torch.Generator().manual_seed(42))
        train_dataset, val_dataset = random_split(train_val_dataset, [
                                                train_len, val_len], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_len = len(full_dataset) - test_len
        train_dataset, test_dataset = random_split(full_dataset, [
                                                train_len, test_len], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = None
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def prepare_dataloaders_per_classes(human_dir, synth_dir, classes, batch_size=256, max_length=70, test_size=0.2, val_size=0.1):
    datasets = []
    for item in classes:
        datasets.append(UnistrokeDataset(
            human_dir=human_dir,
            synth_dir=synth_dir,
            target_class=item,
            max_length=max_length
        ))
        
    full_dataset = ConcatDataset(datasets)

    test_len = int(len(full_dataset) * test_size)
    if val_size != 0:
        train_val_len = len(full_dataset) - test_len
        val_len = int(train_val_len * val_size)
        train_len = train_val_len - val_len

        train_val_dataset, test_dataset = random_split(full_dataset, [
            train_len + val_len, test_len], generator=torch.Generator().manual_seed(42))
        train_dataset, val_dataset = random_split(train_val_dataset, [
            train_len, val_len], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_len = len(full_dataset) - test_len
        train_dataset, test_dataset = random_split(full_dataset, [
            train_len, test_len], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = None
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
