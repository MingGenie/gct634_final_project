import numpy as np
import os
import librosa
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from model import DemoModel

def data_preparation():
    if os.path.isfile('gtzan.tar.gz'):
        pass
    else:
        os.system('gdown --id 1J1DM0QzuRgjzqVWosvPZ1k7MnBRG-IxS')
        os.system('tar zxf gtzan.tar.gz')
    train = load_split('gtzan/split/train.txt')
    test = load_split('gtzan/split/test.txt')
    for genre in genres:
        os.makedirs('gtzan/spec/' + genre, exist_ok=True)
    for path_in in tqdm(train + test):
        # The spectrograms will be saved under `gtzan/spec/` with an file extension of `.npy`
        path_out = 'gtzan/spec/' + path_in.replace('.wav', '.npy')

        # Skip if the spectrogram already exists
        if os.path.isfile(path_out):
            continue

        # Load the audio signal with the desired sampling rate (SR).
        sig, _ = librosa.load(f'gtzan/wav/{path_in}', sr=SR, res_type='kaiser_fast')
        # Compute power mel-spectrogram.
        melspec = librosa.feature.melspectrogram(sig, sr=SR, n_fft=FFT_SIZE, hop_length=FFT_HOP, n_mels=NUM_MELS)
        # Transform the power mel-spectrogram into the log compressed mel-spectrogram.
        melspec = librosa.power_to_db(melspec)
        # "float64" uses too much memory! "float32" has enough precision for spectrograms.
        melspec = melspec.astype('float32')

        # Save the spectrogram.
        np.save(path_out, melspec)


def load_split(path):
    with open(path) as f:
        paths = [line.rstrip('\n') for line in f]
    return paths


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

def accuracy(source, target):
    source = source.max(1)[1].long().cpu()
    target = target.cpu()
    correct = (source == target).sum().item()
    return correct / float(source.shape[0])

def main(args):
    # Data setup
    global SR, FFT_HOP, FFT_SIZE, NUM_MELS, BATCH_SIZE, LR, MOMENTUM, NUM_EPOCHS, WD
    SR = 16000
    FFT_HOP = 512
    FFT_SIZE = 1024
    NUM_MELS = 96
    BATCH_SIZE = 4

    # Training setup.
    LR = 0.0006  # learning rate
    MOMENTUM = 0.9
    NUM_EPOCHS = 10
    WD = 0.0  # L2 regularization weight

    data_preparation()

    train = load_split('gtzan/split/train.txt')
    test = load_split('gtzan/split/test.txt')
    # Load all spectrograms.
    dataset_train = SpecDataset(train)
    specs = [s for s, _ in dataset_train]
    # Compute the minimum temporal dimension size.
    time_dims = [s.shape[1] for s in specs]
    min_time_dim_size = min(time_dims)
    # Stack the spectrograms
    specs = [s[:, :min_time_dim_size] for s in specs]
    specs = np.stack(specs)
    # Compute mean and standard deviation for standard normalization.
    mean = specs.mean()
    std = specs.std()

    dataset_train = SpecDataset(train, mean, std, min_time_dim_size)
    dataset_test = SpecDataset(test, mean, std, min_time_dim_size)

    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True)
    loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False,
                             drop_last=False)

    model = DemoModel(input_dim=NUM_MELS, output_dim=len(genres))

    # Define a loss function, which is cross entropy here.
    criterion = torch.nn.CrossEntropyLoss()
    # Setup an optimizer. Here, we use Stochastic gradient descent (SGD) with a nesterov mementum.
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, nesterov=True, weight_decay=WD)
    # Choose a device. We will use GPU if it's available, otherwise CPU.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Move variables to the desired device.
    model.to(device)
    criterion.to(device)

    print(f'Optimizer: {optimizer}')
    print(f'Device: {device}')

    train_model(model, criterion, optimizer, loader_train,len(dataset_train), device)
    eval_model(model, criterion, loader_test,len(dataset_test), device)


def train_model(model, criterion, optimizer, loader_train, len_data, device):
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        epoch_acc = 0
        pbar = tqdm(loader_train, desc=f'Epoch {epoch:02}')  # progress bar
        for x, y in pbar:
            # Move mini-batch to the desired device.
            x = x.to(device)
            y = y.to(device)

            # Feed forward the model.
            prediction = model(x)
            # Compute the loss.
            loss = criterion(prediction, y)
            # Compute the accuracy.
            acc = accuracy(prediction, y)

            # Perform backward propagation to compute gradients.
            loss.backward()
            # Update the parameters.
            optimizer.step()
            # Reset the computed gradients.
            optimizer.zero_grad()

            # Log training metrics.
            batch_size = len(x)
            epoch_loss += batch_size * loss.item()
            epoch_acc += batch_size * acc
            # Update the progress bar.
            pbar.set_postfix({'loss': epoch_loss / len_data,
                              'acc': epoch_acc / len_data})


def eval_model(model, criterion, loader_test, len_data, device):
    # Set the status of the model as evaluation.
    model.eval()

    # `torch.no_grad()` disables computing gradients. The gradients are still
    # computed even though you use `model.eval()`. You should use `torch.no_grad()`
    # if you don't want your memory is overflowed because of unnecesary gradients.
    with torch.no_grad():
        epoch_loss = 0
        epoch_acc = 0
        pbar = tqdm(loader_test, desc=f'Test')  # progress bar
        for x, y in pbar:
            # Move mini-batch to the desired device.
            x = x.to(device)
            y = y.to(device)

            # Feed forward the model.
            prediction = model(x)
            # Compute the loss.
            loss = criterion(prediction, y)
            # Compute the accuracy.
            acc = accuracy(prediction, y)

            # Log training metrics.
            batch_size = len(x)
            epoch_loss += batch_size * loss.item()
            epoch_acc += batch_size * acc
            # Update the progress bar.
            pbar.set_postfix({'loss': epoch_loss / len_data, 'acc': epoch_acc / len_data})

    # Compute the evaluation scores.
    test_loss = epoch_loss / len_data
    test_acc = epoch_acc / len_data

    print(f'test_loss={test_loss:.5f}, test_acc={test_acc * 100:.2f}%')


def parse():
    parser = argparse.ArgumentParser(description='GCT634 final project demo python')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()
    if not torch.cuda.is_available():
        raise SystemError('GPU device not found!')
    print(f'Found GPU at: {torch.cuda.get_device_name()}')
    print(f'PyTorch version: {torch.__version__}')
    print(f'Librosa version: {librosa.__version__}')

    global genres, genre_dict
    genres = genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae']
    genre_dict = {g: i for i, g in enumerate(genres)}
    main(args)