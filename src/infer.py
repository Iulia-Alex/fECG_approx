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

def get_model(path, sizes=(128, 128)):
    model = ComplexUNet(sizes[0] * sizes[1])
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def load_mat(path):
    data = loadmat(path)
    mecg = data['out']['mecg'][0][0].astype(float)
    fecg = data['out']['fecg'][0][0].astype(float)
    
    resampler = torchaudio.transforms.Resample(orig_freq=2500, new_freq=500)
    mecg = resampler(torch.tensor(mecg, dtype=torch.float32))
    fecg = resampler(torch.tensor(fecg, dtype=torch.float32))
    
    random_start = torch.randint(0, mecg.size(-1) - 1915, (1,))
    mecg = mecg[:, random_start:random_start+1915]
    fecg = fecg[:, random_start:random_start+1915]
    
    return mecg, fecg


@click.command()
@click.option('-m', '--model_path', type=str, default='models/best.pth')
@click.option('-s', '--signal_path', type=str, default='data/ecg/fecgsyn01.mat')
@click.option('--snr_db', type=int, default=20)
def main(model_path, signal_path, snr_db):
    model = get_model(model_path).to('cuda')
    stft = STFT()
    diffuser = Diffuser(4, 500)
    
    mecg, fecg = load_mat(signal_path)
    sum_ = diffuser(mecg + fecg, snr_db)
    
    sum_spec = stft.stft_only(sum_).to('cuda')
    sum_spec = sum_spec[:, :-1, :]  # drop last freq bin
    sum_spec = sum_spec.unsqueeze(0)  # add batch dimension
    sum_spec = sum_spec / 10.0
    
    with torch.no_grad():
        pred = model(sum_spec)[0]
    
    pred = pred.detach().to('cpu')
    pred = pred * 10.0
    b, f, t = pred.shape
    pred = torch.cat([pred, torch.zeros(b, 1, t, dtype=pred.dtype, device=pred.device)], dim=1)
    pred = stft.istft_only(pred, length=1915).numpy()
    
    plt.figure(figsize=(12, 6))
    plt.plot(fecg[0], label='true')
    plt.plot(pred[0], '--r', label='pred')
    plt.legend()
    plt.savefig('results/result.png')


if __name__ == '__main__':
    main()
   
    
    
    
    
       











