import os
import click
import torch
import torchaudio
from scipy.io import loadmat

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from fourier import STFT
from network import ComplexUNet
from diffuser import Diffuser


def get_model(path, sizes, sameW, activation, diag):
    model = ComplexUNet(sizes[0] * sizes[1], sameW=sameW, activation=activation, diag=diag)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def load_mat(path, random=False):
    data = loadmat(path)
    mecg = data['out']['mecg'][0][0].astype(float)
    fecg = data['out']['fecg'][0][0].astype(float)
    
    resampler = torchaudio.transforms.Resample(orig_freq=2500, new_freq=500)
    mecg = torch.tensor(mecg, dtype=torch.float32)
    fecg = torch.tensor(fecg, dtype=torch.float32)
    # mecg = mecg / mecg.abs().max()
    # fecg = fecg / fecg.abs().max()
    mecg = resampler(mecg)
    fecg = resampler(fecg)
    
    if random:
        random_start = torch.randint(0, len(mecg) - 1915, (1,)).item()
    else:
        random_start = 42 * 500
    mecg = mecg[:, random_start:random_start+1915]
    fecg = fecg[:, random_start:random_start+1915]
    
    return mecg, fecg


@click.command()
@click.option('-m', '--model_path', type=str, default='models/best.pth')
@click.option('-s', '--signal_path', type=str, default='data/test_ecg2')
### examples of mat from test: 23, 542, 598, 616,
@click.option('--snr_db', type=int, default=5)
@click.option('-i', '--index', type=int, default=1)
def main(model_path, signal_path, snr_db, index):
    model = get_model(model_path, sizes=(128, 128), sameW=False, activation='crelu', diag=False)
    model = model.to('cuda:1')
    stft = STFT()
    diffuser = Diffuser(500)
    
    index = str(index).zfill(2)
    signal_path = os.path.join(signal_path, f'fecgsyn{index}.mat')
    
    mecg, fecg = load_mat(signal_path)
    sum_ = diffuser(mecg + fecg, snr_db)
    # sum_[:, 0] = 1.0
    
    sum_spec = stft.stft(sum_).to('cuda:1')
    sum_spec = sum_spec[:, :-1, :]  # drop last freq bin
    sum_spec = sum_spec.unsqueeze(0)  # add batch dimension
    # sum_spec = sum_spec / 10.0
    
    with torch.no_grad():
        pred = model(sum_spec)[0]
    
    pred = pred.detach().to('cpu')
    # pred = pred * 10.0
    b, f, t = pred.shape
    pred = torch.cat([pred, torch.zeros(b, 1, t, dtype=pred.dtype, device=pred.device)], dim=1)
    pred = stft.istft(pred, length=1915).numpy()
    
    fig, ax = plt.subplots(2, 2, figsize=(24, 12))
    for i in range(4):
        ax[i // 2, i % 2].plot(fecg[i], label='true')
        ax[i // 2, i % 2].plot(pred[i], '--r', label='pred')
        ax[i // 2, i % 2].legend(loc='upper right')
        ax[i // 2, i % 2].set_title(f"Channel {i+1}")
    plt.suptitle(f"Prediction for {signal_path}", fontsize=16)
    plt.tight_layout()
    
    output_path = f'results/result_{index}.png'
    plt.savefig(output_path)
    plt.close()


if __name__ == '__main__':
    main()
   
    
    
    
    
       











