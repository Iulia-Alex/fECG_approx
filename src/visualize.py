import click
import torch
import torchaudio

import pandas as pd

from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')



# init_sr = int(5e6 / 256 / 64)



def load_mat(path):
    init_sr = 2500
    final_sr = 500
    resampler = torchaudio.transforms.Resample(init_sr, final_sr)
    
    pack = loadmat(path)
    print(pack.keys())
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
    return ecg


def plot(ecg, path):
    
    # ecg = ecg[:, 42*500:42*500+1915]
    
    fig, ax = plt.subplots(2, 2, figsize=(24, 12))
    for i in range(4):
        ax[i // 2, i % 2].plot(ecg[i], label='true')
        ax[i // 2, i % 2].legend(loc='upper right')
        ax[i // 2, i % 2].set_title(f"Channel {i+1}")
    plt.suptitle(f"Prediction for {path}", fontsize=16)
    plt.tight_layout()
    
    output_path = f'results/result.png'
    plt.savefig(output_path)
    plt.close()


    
    
@click.command()
@click.option('--signal_path', type=str, help='Path to the predicted signals', default='data/predicted_ecg_before_ro/fecgsyn')
@click.option('--idx', type=int, help='Index of the signal', default=1)
def main(signal_path, idx):
    
    # path = './data/Data EWAM/EWAM1/#4/Data_Acq4/T3_1.mat'
    path = './data/Data EWAM/EWAM1/#3/S3_2.mat'  # this seems to be ok?
    path = './data/Data EWAM/EWAM1/#2/t5.mat'
    path = './data/Data EWAM/EWAM1/#2/Data1.mat'
    path = './data/Data EWAM/EWAM1/#2/t5.mat'
    # path = './data/Data EWAM/EWAM1/#2/t6_wet.mat'
    # ecg = load_mat(path)
    
    
    # path = './data/Data EWAM/EWAM4/#5/vol5_dry2.16ch'
    path ='./data/Data EWAM/EWAM4/#5/vol5_dry1.16ch'
    ecg = load_dry(path)
    
    
    plot(ecg, path)
    
    
if __name__ == '__main__':
    main()
    
    
    
