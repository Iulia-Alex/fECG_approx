import os
import torch
import torchaudio
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('darkgrid')

resampler = torchaudio.transforms.Resample(orig_freq=1000, new_freq=500)

def load_mat(path):
    ecg = loadmat(path)
    ecg_signal = ecg['ecg'].T
    ecg_signal = resampler(torch.tensor(ecg_signal, dtype=torch.float32))
    peaks = ecg['peaks'][0]
    peaks = torch.tensor(peaks, dtype=torch.long) // 2  # 1000 Hz to 500 Hz changed positions
    return ecg_signal, peaks



if __name__ == '__main__':
    fname = '/home/madenn2/Documents/phd/fECG_approx/data/physio_setA/a03.mat'
    
    ecg, peaks = load_mat(fname)
    
    cut = 600
    ecg = ecg[:, :cut]
    # smoothed_ecg = smoothed_ecg[:, :cut]
    peaks = [int(p) for p in peaks if p < cut]
    print(ecg.shape, peaks[-1])

    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    for i in range(4):
        ax[i // 2, i % 2].plot(ecg[i], color='blue', label='original')
        # ax[i // 2, i % 2].scatter(peaks, ecg[i][peaks], color='red', label='peaks')
        # ax[i // 2, i % 2].plot(smoothed_ecg[i], color='green', label='generated')
        # ax[i // 2, i % 2].plot(mask[i], color='black')
        # ax[i // 2, i % 2].legend(loc='lower right')
    plt.suptitle(f'Simulated fECG for PhysioNet dataset, file {os.path.basename(fname)}')
    plt.tight_layout()
    plt.savefig('ecg.png')