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
from network import create_model
from diffuser import Diffuser
from metrics import PDR
from loss import SignalMSE

  

def load_mat(path, random=False):
    orig_sr = 2500
    new_sr = 500
    resampler = torchaudio.transforms.Resample(orig_sr, new_sr)
    
    data = loadmat(path)
    mecg = data['out']['mecg'][0][0].astype(float)
    fecg = data['out']['fecg'][0][0].astype(float)

    mecg = resampler(torch.tensor(mecg, dtype=torch.float32))
    fecg = resampler(torch.tensor(fecg, dtype=torch.float32))
    return mecg, fecg


def replace_nans(signal):
    nans = torch.isnan(signal)
    nans_idx = torch.where(nans)[0]
    for i in nans_idx:
        if i == 0:
            signal[i] = signal[i+1] if not torch.isnan(signal[i+1]) else 0
        else :
            signal[i] = signal[i-1]
    return signal


def load_mat_physio(path):
    orig_sr = 1000
    new_sr = 500
    resampler = torchaudio.transforms.Resample(orig_sr, new_sr)
    
    data = loadmat(path)
    ecg = data['ecg'].astype(float)
    ecg = torch.tensor(ecg, dtype=torch.float32).T
    for i in range(ecg.shape[0]):
        ecg[i] = replace_nans(ecg[i])
    real_peaks = data['peaks'].astype(int)
    real_peaks = list(real_peaks[0])
    ecg = resampler(torch.tensor(ecg, dtype=torch.float32))
    real_peaks = [int(p * (new_sr / orig_sr)) for p in real_peaks]
    real_peaks = torch.tensor(real_peaks, dtype=torch.long)
    return ecg, real_peaks


def load_mat_iulia(path, size=1915):
    orig_sr = 2500
    new_sr = 500
    resampler = torchaudio.transforms.Resample(orig_sr, new_sr)
    
    data = loadmat(path)
    signal = data['s'].astype(float)
    signal = torch.tensor(signal, dtype=torch.float32)
    signal = signal.view(1, -1)
    signal = resampler(signal)
    return signal[:, :size]



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
    
    signals_path = 'data/test_ecg2'
    signals_path = 'data/iulia_test_files'
    save_signals_path = f'results/iulia_test_files'
    
    snr_db = [5, 20]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    ckpt_path = 'models/latest_model_metadata.pth'
    model = create_model(ckpt_path)
    model = model.to(device)
    model = model.eval()
    
    diffuser = Diffuser(500, snr_db=snr_db)
    stft = STFT()
    loss_fn = SignalMSE(stft)
    metric_fn = PDR(stft)
    
    
    files = sorted(os.listdir(signals_path))
    os.makedirs(save_signals_path, exist_ok=True)
    
    cnt_nans = 0
    mse_list = []
    pdr_list = []
    for file in tqdm(files):
        signal_path = os.path.join(signals_path, file)
    
        # mecg, fecg = load_mat(signal_path)
        # sum_ = mecg + fecg
        # sum_ = diffuser(sum_)
        # sum_, peaks = load_mat_physio(signal_path)
        sum_ = load_mat_iulia(signal_path)

        batches = create_batches_from_signal(sum_, stft)
        pred_batches = []
        
        for batch in batches:
            batch = batch.unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(batch)
            pred_batches.append(pred)
            
        pred = torch.cat(pred_batches, dim=0).to('cpu')
        
        signal_pred = create_signal_from_batches(pred, stft, sum_.shape[-1])
        
        # mse = loss_fn(fecg, signal_pred, signal=True).item()
        # pdr = metric_fn(fecg, signal_pred, signal=True)
        # mse_list.append(mse)
        # pdr_list.append(pdr['prd'])
        # tqdm.write(f'File: {file}, MSE: {mse}, PDR: {pdr["prd"]}')
        tqdm.write(f"File: {file} done!")

        # compute the nans in the signal
        num_nans = torch.isnan(sum_).sum().item()
        if num_nans > 0:
            tqdm.write(f'Found {num_nans} nans in the signal {file}')
            cnt_nans += 1
            continue

        
        signals = {
            # 'original_fecg': fecg.numpy(),
            # 'original_mecg': mecg.numpy(),
            'predicted_fecg': signal_pred.numpy(),
            'noisy_signal': sum_.numpy(),
            # 'peaks': peaks.numpy()
        }
        
        save_path = os.path.join(save_signals_path, file)
        savemat(save_path, signals)
        # tqdm.write(f'Saved {save_path} with {signal_pred.shape[-1]} samples, MSE: {mse}, PDR: {pdr["prd"]}')
        # tqdm.write(f'Saved {save_path} with {signal_pred.shape[-1]} samples and {peaks.shape[0]} peaks')


    mse_list = torch.tensor(mse_list)
    pdr_list = torch.tensor(pdr_list)
    
    mse_mean = mse_list.mean().item()
    mse_std = mse_list.std().item()
    
    pdr_mean = pdr_list.mean().item()
    pdr_std = pdr_list.std().item()
    
    print(f'Found {cnt_nans} signals with nans')
    print(f'MSE: {mse_mean} +- {mse_std}')
    print(f'PDR: {pdr_mean} +- {pdr_std}')
    

        
        

    
    
    



    

    
    
    
    
    