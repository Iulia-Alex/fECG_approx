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
from metrics import PDR
from loss import SignalMSE


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


def create_batches_from_signal(signal, stft, samples_size=1915, overlap=0.5):
    batches = []
    for i in range(0, signal.shape[-1], int(overlap * samples_size)):
        spec = stft.stft(signal[:, i:i+samples_size])
        spec = spec[:, :-1, :]  # drop last freq bin
        
        if spec.shape[-1] < 128:
            spec = torch.cat([spec, torch.zeros(spec.shape[0], spec.shape[1], 128 - spec.shape[-1])] , dim=-1)
        
        batches.append(spec)
    return torch.stack(batches)


def create_signal_from_batches(specs, stft, original_size, samples_size=1915, overlap=0.5):
    b, c, f, t = specs.shape
    zeros = torch.zeros(b, c, 1, t, dtype=pred.dtype)
    specs = torch.cat([specs, zeros], dim=2)
    signal = torch.zeros(4, specs.shape[0] * samples_size)
    for i, spec in enumerate(specs):
        recovered_signal = stft.istft(spec, length=samples_size)
        recovered_signal *= torch.hann_window(samples_size)
        start = i * int(overlap * samples_size)
        end = start + samples_size
        signal[:, start:end] += recovered_signal
        
    return signal[:, :original_size]



if __name__ == '__main__':
    # model_path = 'models/unet_sameW_snr15.pth'
    # model_path = 'models/BEST_unet_norm_nodiag_diffW_snr5.pth'
    # model_path = 'models/unet_norm_nodiag_diffW_snr5.pth'
    model_path = 'models/best_v2_noDatasetNorm.pth'
    signals_path = 'data/test_ecg2'
    snr_db = 5
    
    model = get_model(model_path).to('cuda:1')
    diffuser = Diffuser(500)
    stft = STFT()
    loss_fn = SignalMSE(stft)
    metric_fn = PDR(stft)
    
    
    files = os.listdir(signals_path)
    # files = ['fecgsyn01.mat']
    
    cnt_nans = 0
    mse_list = []
    pdr_list = []
    for file in tqdm(files):
        signal_path = os.path.join(signals_path, file)
    
        mecg, fecg = load_mat(signal_path)
        sum_ = mecg + fecg
        sum_ = diffuser(sum_, snr_db)
        # sum_ = sum_ / sum_.abs().max()
        
        
        batches = create_batches_from_signal(sum_, stft)
        # batches = batches / 10.0
        
        with torch.no_grad():
            pred = model(batches.to('cuda:1'))
            
        pred = pred.cpu()
        # pred = pred * 10.0
        
        signal_pred = create_signal_from_batches(pred, stft, sum_.shape[-1])
        
        mse = loss_fn(fecg, signal_pred, signal=True).item()
        pdr = metric_fn(fecg, signal_pred, signal=True)
        mse_list.append(mse)
        pdr_list.append(pdr['prd'])

        # compute the nans in the signal
        num_nans = torch.isnan(signal_pred).sum().item()
        if num_nans > 0:
            tqdm.write(f'Found {num_nans} nans in the signal {file}')
            cnt_nans += 1
            continue

        
        signals = {
            'original_fecg': fecg.numpy(),
            'original_mecg': mecg.numpy(),
            'predicted_fecg': signal_pred.numpy(),
            'noisy_signal': sum_.numpy()
        }
        
        save_path = signal_path.replace('test_ecg', 'predicted_ecg')
        # savemat(save_path, signals)
    
    
    mse_list = torch.tensor(mse_list)
    pdr_list = torch.tensor(pdr_list)
    
    mse_mean = mse_list.mean().item()
    mse_std = mse_list.std().item()
    
    pdr_mean = pdr_list.mean().item()
    pdr_std = pdr_list.std().item()
    
    print(f'Found {cnt_nans} signals with nans')
    print(f'MSE: {mse_mean} +- {mse_std}')
    print(f'PDR: {pdr_mean} +- {pdr_std}')
    

        
        

    
    
    



    

    
    
    
    
    