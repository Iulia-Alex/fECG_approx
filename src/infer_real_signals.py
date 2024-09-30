import os
import click
import torch
import torchaudio
from tqdm import tqdm
from scipy.io import loadmat, savemat

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from fourier import STFT
from network import ComplexUNet
from diffuser import Diffuser


def get_model(path, sizes=(128, 128)):
    model = ComplexUNet(sizes[0] * sizes[1], sameW=False)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def load_mat(path, random=False):
    data = loadmat(path)
    mecg = data['out']['mecg'][0][0].astype(float)
    fecg = data['out']['fecg'][0][0].astype(float)
        
    resampler = torchaudio.transforms.Resample(orig_freq=2500, new_freq=500)
    mecg = resampler(torch.tensor(mecg, dtype=torch.float32))
    fecg = resampler(torch.tensor(fecg, dtype=torch.float32))
    return mecg, fecg


def create_batches_from_signal(signal, stft, samples_size=1915):
    batches = []
    for i in range(0, signal.shape[-1], int(0.75 * samples_size)):
        spec = stft.stft_only(signal[:, i:i+samples_size])
        spec = spec[:, :-1, :]  # drop last freq bin
        
        if spec.shape[-1] < 128:
            spec = torch.cat([spec, torch.zeros(spec.shape[0], spec.shape[1], 128 - spec.shape[-1])] , dim=-1)
        
        batches.append(spec)
    return torch.stack(batches)


def create_signal_from_batches(specs, stft, original_size, samples_size=1915):
    b, c, f, t = specs.shape
    zeros = torch.zeros(b, c, 1, t, dtype=pred.dtype)
    specs = torch.cat([specs, zeros], dim=2)
    signal = torch.zeros(4, specs.shape[0] * samples_size)
    for i, spec in enumerate(specs):
        recovered_signal = stft.istft_only(spec, length=samples_size)
        recovered_signal *= torch.hann_window(samples_size)
        start = i * int(0.75 * samples_size)
        end = start + samples_size
        signal[:, start:end] += recovered_signal
        
    return signal[:, :original_size]



if __name__ == '__main__':
    # model_path = 'models/unet_sameW_snr15.pth'
    model_path = 'models/best_unet_nodiag_diffW_snr5.pth'
    signals_path = 'data/test_ecg'
    snr_db = 15
    
    model = get_model(model_path).to('cuda')
    
    
    
    files = os.listdir(signals_path)
    # files = ['fecgsyn01.mat']
    
    cnt_nans = 0
    
    for file in tqdm(files):
        signal_path = os.path.join(signals_path, file)
    
        mecg, fecg = load_mat(signal_path)
        sum_ = Diffuser(4, 500)(mecg + fecg, snr_db)
        sum_[:, 0] = 0.0  # first sample is always 0
        stft = STFT()
        
        batches = create_batches_from_signal(sum_, stft)
        batches = batches / 10.0
        
        with torch.no_grad():
            pred = model(batches.to('cuda'))
            
        pred = pred.cpu()
        pred = pred * 10.0
        
        
        signal_pred = create_signal_from_batches(pred, stft, sum_.shape[-1])
        
        # print(fecg.shape, signal_pred.shape)
        
        # compute the nans in the signal
        # num_nans = torch.isnan(signal_pred).sum().item()
        # if num_nans > 0:
        #     print(f'Found {num_nans} nans in the signal {file}')
        #     cnt_nans += 1
        #     continue

        
        signals = {
            'original_fecg': fecg.numpy(),
            'original_mecg': mecg.numpy(),
            'predicted_fecg': signal_pred.numpy(),
            'noisy_signal': sum_.numpy()
        }
        
        save_path = signal_path.replace('test_ecg', 'predicted_ecg')
        # savemat(save_path, signals)
    
    print(f'Found {cnt_nans} signals with nans')
    

        
        

    
    
    



    

    
    
    
    
    