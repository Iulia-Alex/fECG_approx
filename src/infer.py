import os
import click
import torch
import torchaudio
from scipy.io import loadmat

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
# disable grid
# plt.rcParams['axes.grid'] = False

from fourier import STFT
from network import ComplexUNet, create_model
from diffuser import Diffuser


def get_model(path, sizes, sameW, activation, diag):
    model = ComplexUNet(sizes[0] * sizes[1], sameW=sameW, activation=activation, diag=diag)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


resampler = torchaudio.transforms.Resample(orig_freq=1000, new_freq=500)


def load_mat(path, random=False):
    data = loadmat(path)
    mecg = data['out']['mecg'][0][0].astype(float)
    fecg = data['out']['fecg'][0][0].astype(float)
    # mecgs = data['ecg'].astype(float)
    
    mecg = torch.tensor(mecg, dtype=torch.float32)
    fecg = torch.tensor(fecg, dtype=torch.float32)
    mecg = mecg / mecg.abs().max()
    fecg = fecg / fecg.abs().max()
    mecg = resampler(mecg)
    fecg = resampler(fecg)
    
    if random:
        random_start = torch.randint(0, len(mecg) - 1915, (1,)).item()
    else:
        random_start = 43 * 500
    mecg = mecg[:, random_start:random_start+1915]
    fecg = fecg[:, random_start:random_start+1915]
    
    return mecg, fecg



def load_mat_EWAM1(path):
    pack = loadmat(path)
    # print(pack.keys())
    s1 = pack['sampledata_input1']
    s2 = pack['sampledata_input2']
    s3 = pack['sampledata_input3']
    s4 = pack['sampledata_input4']
    # s1 = pack['ADC1sat']
    # s2 = pack['ADC2sat']
    # s3 = pack['ADC3sat']
    # s4 = pack['ADC4sat']
    
    sum_ = torch.zeros(4, s1.shape[0])
    sum_[0] = torch.tensor(s1).squeeze()
    sum_[1] = torch.tensor(s2).squeeze()
    sum_[2] = torch.tensor(s3).squeeze()
    sum_[3] = torch.tensor(s4).squeeze()
    
    sum_ = resampler(sum_)
    
    sum_ = sum_[:, 42*500:42*500+1915]
    
    return sum_
    
def load_dry(path):
    init_sr = int(5e6 / 256 / 64)
    final_sr = 500
    resampler = torchaudio.transforms.Resample(init_sr, final_sr)
    
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line.split() for line in lines]
    lines = [[float(x) for x in line] for line in lines]
    lines = torch.tensor(lines)
    
    Vfs_ADC = 2 #; %Vpp
    VAgain = 5.5 #; %gain setting of the VA
    SDCgain = 1 #; %gain setting of the SDC
    Vfs_input = Vfs_ADC / VAgain / SDCgain / 4 #; %extra factor 4 because of 2x gain of 2nd stage amp and inherent 2x in SDC
    Vfs_coderange = 2 ^ 14 #; %Nominal code range of the ADC
    decimationfactor = 64 #;
    GainLSBperV = Vfs_coderange / Vfs_input * decimationfactor;
    
    lines = lines / GainLSBperV

    ecg = resampler(lines)
    
    ecg = ecg[:, 42*500:42*500+1915]
    
    return ecg



def replace_nans(signal):
    nans = torch.isnan(signal)
    nans_idx = torch.where(nans)[0]
    for i in nans_idx:
        if torch.isnan(signal[i]) and torch.isnan(signal[i+1]):
            print(f'2 nans in a row at position {i} and {i+1}')
        
        if i == 0:
            signal[i] = signal[i+1] if not torch.isnan(signal[i+1]) else 0
        else :
            signal[i] = signal[i-1]
    return signal, nans_idx


def load_mat_physio(path):
    orig_sr = 1000
    new_sr = 500
    resampler = torchaudio.transforms.Resample(orig_sr, new_sr)
    
    data = loadmat(path)
    ecg = data['ecg'].astype(float)
    ecg = torch.tensor(ecg, dtype=torch.float32).T
    nans_pos_list = []
    for i in range(ecg.shape[0]):
        print("Analyzing channel", i, "...")
        ecg[i], nans_pos = replace_nans(ecg[i])
        nans_pos_list.append(nans_pos)
    real_peaks = data['peaks'].astype(int)
    real_peaks = list(real_peaks[0])
    ecg = resampler(torch.tensor(ecg, dtype=torch.float32))
    real_peaks = [int(p * (new_sr / orig_sr)) for p in real_peaks]
    # nans_pos_list = [[int(p * (new_sr / orig_sr)) for p in nans_pos] for nans_pos in nans_pos_list]
    real_peaks = torch.tensor(real_peaks, dtype=torch.long)
    return ecg, real_peaks, nans_pos_list




@click.command()
@click.option('-m', '--model_path', type=str, default='models/best.pth')
@click.option('-s', '--signal_path', type=str, default='data/test_ecg/fecgsyn01.mat')
### examples of mat from test: 23, 542, 598, 616,
@click.option('--snr_db', type=int, default=5)
@click.option('-i', '--index', type=int, default=1)
def main(model_path, signal_path, snr_db, index):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = 'models/latest_model_metadata.pth'
    # model = get_model(model_path, sizes=(128, 128), sameW=False, activation='ro', diag=True)
    model = create_model(model_path)
    model = model.to(device)
    stft = STFT()
    diffuser = Diffuser(500, snr_db=20)
    
    # index = str(index).zfill(2)
    # signal_path = os.path.join(signal_path, f'fecgsyn{index}.mat')
    # signal_path = os.path.join(signal_path, f'a{index}.mat')
    
    mecg, fecg = load_mat(signal_path)
    sum_ = diffuser(mecg + fecg)
    # sum_, _ = load_mat(signal_path, random=True)
    
    
    # signal_path = './data/Data EWAM/EWAM1/#2/t5.mat'
    # signal_path = './data/Data EWAM/EWAM1/#2/t6_wet.mat'
    # sum_ = load_mat_EWAM1(signal_path)
    
    
    # signal_path = './data/Data EWAM/EWAM4/#5/vol5_dry1.16ch'
    # sum_ = load_dry(signal_path)
    
    
    # signal_path = './data/physio_setA/a18.mat'
    # sum_, peaks, nans_pos = load_mat_physio(signal_path)
    
    # max_consecutive_nans = 0
    # for nans in nans_pos:
    #     tmp_max = 0
    #     for i in range(len(nans)-1):
    #         if nans[i+1] - nans[i] == 1:
    #             tmp_max += 1
    #         else:
    #             if tmp_max > max_consecutive_nans:
    #                 max_consecutive_nans = tmp_max
    #             tmp_max = 0
    # print(f"Max consecutive nans: {max_consecutive_nans}")
    
    # fig, ax = plt.subplots(2, 2, figsize=(24, 12))
    # for i in range(4):
    #     ax[i // 2, i % 2].plot(sum_[i], label='signal')
    #     ax[i // 2, i % 2].set_title(f"Channel {i+1}")
    #     ax[i // 2, i % 2].scatter(nans_pos[i], sum_[i][nans_pos[i]], c='r', label='nans')
    #     ax[i // 2, i % 2].legend(loc='upper right')
    
    # plt.suptitle(f"Physio Set A: {signal_path}", fontsize=16)
    # plt.tight_layout()
    # plt.savefig('results/physio_setA_nans.png')

    print("sum_ shape:", sum_.shape)
    print("spec shape:", stft.stft(sum_).to(device).shape)
    
    sum_spec = stft.stft(sum_[0:1]).to(device)
    sum_spec = sum_spec[:, :-1, :]  # drop last freq bin
    sum_spec = sum_spec.unsqueeze(0)  # add batch dimension
    # sum_spec = sum_spec / 10.0
    sum_spec = sum_spec.contiguous()
    
    with torch.no_grad():
        pred = model(sum_spec)[0]
    
    pred_spec = pred.detach().to('cpu')
    # pred = pred * 10.0
    b, f, t = pred_spec.shape
    pred = torch.cat([pred_spec, torch.zeros(b, 1, t, dtype=pred_spec.dtype, device=pred_spec.device)], dim=1)
    pred = stft.istft(pred, length=1915).numpy()
    

    # fecg = fecg.numpy()
    fecg_spec = stft.stft(fecg).to(device)
    fecg_spec = fecg_spec[:, :-1, :]  # drop last freq bin



    fig, ax = plt.subplots(2, 2, figsize=(24, 12))
    for i in range(1):
        if i == 1:
            ax[i // 2, i % 2].plot(sum_[i], label='true', c='g', linewidth=5)
            ax[(2 + i) // 2, (i + 2) % 2].imshow(sum_spec[0, i].abs().log1p().cpu().numpy(), aspect='auto', origin='lower')
        if i == 1:
            ax[(i -1) // 2, (i - 1) % 2].plot(fecg[i], label='true', c='g', linewidth=5)
            ax[(i +1) // 2, (i + 1) % 2].imshow(fecg_spec[i].abs().log1p().cpu().numpy(), aspect='auto', origin='lower')
        # ax[i // 2, i % 2].plot(pred[i], '--r', label='pred')
        # ax[i // 2, i % 2].legend(loc='upper right')
        # ax[i // 2, i % 2].set_title(f"Channel {i+1}")
    plt.suptitle(f"Prediction for {signal_path}", fontsize=16)
    plt.tight_layout()
    
    output_path = f'ecg.png'
    plt.savefig(output_path)
    plt.close()


if __name__ == '__main__':
    main()
   
    
    
    
    
       











