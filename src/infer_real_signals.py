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
from metrics import MeticEvaluator
from loss import SignalMSE, SignalMAE

import sys
  

def load_mat(path):
    orig_sr = 2500
    new_sr = 500
    resampler = torchaudio.transforms.Resample(orig_sr, new_sr)
    
    data = loadmat(path)
    mecg = data['out']['mecg'][0][0].astype(float)
    fecg = data['out']['fecg'][0][0].astype(float)

    mecg = resampler(torch.tensor(mecg, dtype=torch.float32))
    fecg = resampler(torch.tensor(fecg, dtype=torch.float32))
    return mecg, fecg


def load_mat_abdominal(path):
    orig_sr = 100
    new_sr = 500
    resampler = torchaudio.transforms.Resample(orig_sr, new_sr)
    
    data = loadmat(path)
    data = data['out']
    sum_ = data['fecg'][0][0].astype(float)
    fecg = data['real'][0][0].astype(float)

    sum_, fecg = sum_.T, fecg.T
    sum_ = resampler(torch.tensor(sum_, dtype=torch.float32))
    fecg = resampler(torch.tensor(fecg, dtype=torch.float32))
    fecg = fecg.view(1, -1).repeat(4, 1)
    return sum_, fecg


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
    
    # signals_path = 'data/test_ecg2'
    signals_path = 'data/test_ecg'
    signals_path = 'data/abdominal_fecg'
    save_signals_path = f'results/test_ecg'
    debug = True
    
    snr_db = [5, 20]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f'Using device: {device}')

    ckpt_path = 'models/best.pth'
    model = create_model(ckpt_path)
    model = model.to(device)
    model = model.eval()
    
    diffuser = Diffuser(500, snr_db=snr_db)
    stft = STFT()
    loss_fn = SignalMAE(stft, normalize=True)
    metrics = MeticEvaluator(stft, metric_list=['pdr', 'pcc'])
    
    
    files = sorted(os.listdir(signals_path))
    os.makedirs(save_signals_path, exist_ok=True)
    
    cnt_nans = 0
    mse_list = []
    pdr_list = []
    pcc_list = []
    for file in tqdm(files):
        signal_path = os.path.join(signals_path, file)
    
        # mecg, fecg = load_mat(signal_path)
        # sum_ = mecg + fecg
        # sum_ = diffuser(sum_)
        # sum_, peaks = load_mat_physio(signal_path)
        # sum_ = load_mat_iulia(signal_path)
        sum_, fecg = load_mat_abdominal(signal_path)
        if debug:
            print(f'SUM: {sum_.shape}, FECG: {fecg.shape}')

        batches = create_batches_from_signal(sum_, stft)
        pred_batches = []
        
        _, channels, _, _ = batches.shape
        for c in range(channels):
            batch = batches[:, c, :, :]
            mini_batch_pred = []
            for i in range(batch.shape[0] // 64):
                mini_batch = batch[i * 64:(i+1) * 64]
                mini_batch = mini_batch.unsqueeze(1).to(device)
                with torch.no_grad():
                    pred = model(mini_batch)
                mini_batch_pred.append(pred)
            mini_batch_pred = torch.cat(mini_batch_pred, dim=0)        
            pred_batches.append(mini_batch_pred)
        pred = torch.cat(pred_batches, dim=1).to('cpu')
        if debug:
            print(pred.shape)
        
            
        # for batch in batches:
        #     batch = batch.unsqueeze(0).to(device)
        #     b, c, f, t = batch.shape
        #     batch = batch.view(b * c, 1, f, t)
        #     with torch.no_grad():
        #         pred = model(batch)
        #     b, c, f, t = pred.shape
        #     pred = pred.view(b // c, c, f, t)
        #     print(pred.shape)
        #     pred_batches.append(pred)
            
            
        # pred = torch.cat(pred_batches, dim=0).to('cpu')
        
        signal_pred = create_signal_from_batches(pred, stft, sum_.shape[-1])
        
        mse = loss_fn(fecg, signal_pred, signal=True).item()
        metric_results = metrics(fecg, signal_pred, signal=True)
        mse_list.append(mse)
        pdr_list.append(metric_results['pdr'])
        pcc_list.append(metric_results['pcc'])
        if debug:
            print(f'File: {file}, MSE: {mse}, PDR: {metric_results["pdr"]}, PCC: {metric_results["pcc"]}')
        # tqdm.write(f"File: {file} done!")

        # compute the nans in the signal
        num_nans = torch.isnan(sum_).sum().item()
        if num_nans > 0:
            if debug:
                print(f'Found {num_nans} nans in the signal {file}')
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
        # savemat(save_path, signals)
        # tqdm.write(f'Saved {save_path} with {signal_pred.shape[-1]} samples, MSE: {mse}, PDR: {pdr["prd"]}')
        # tqdm.write(f'Saved {save_path} with {signal_pred.shape[-1]} samples and {peaks.shape[0]} peaks')


    mse_list = torch.tensor(mse_list)
    pdr_list = torch.tensor(pdr_list)
    
    mse_mean = mse_list.mean().item()
    mse_std = mse_list.std().item()
    
    pdr_mean = pdr_list.mean().item()
    pdr_std = pdr_list.std().item()
    
    pcc_mean = torch.tensor(pcc_list).mean().item()
    pcc_std = torch.tensor(pcc_list).std().item()
    
    print(f'Found {cnt_nans} signals with nans')
    print(f'MSE: {mse_mean} +- {mse_std}')
    print(f'PDR: {pdr_mean} +- {pdr_std}')
    print(f'PCC: {pcc_mean} +- {pcc_std}')
    

        
        

    
    
    



    

    
    
    
    
    