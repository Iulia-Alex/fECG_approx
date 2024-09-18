import os
import torch
import torchaudio
from scipy.io import loadmat

from diffuser import Diffuser
from fourier import STFT


class SignalDataset(torch.utils.data.Dataset):
    def __init__(self, folder, seconds_per_sample=3.83, channels=4, snr_db=10):
        self.files = self.__get_files(folder)
        resample_fn = torchaudio.transforms.Resample(orig_freq=2500, new_freq=500)
        self.samples = int(seconds_per_sample * 500)
        self.diffuser = Diffuser(num_channels=channels, sample_rate=500)
        self.transform = lambda x: self.__get_random_samples(resample_fn(torch.tensor(x, dtype=torch.float32) / torch.tensor(x).abs().max()))
        self.snr_db = snr_db
        self.random_cut = True
        self.stft = STFT()
        

    def __len__(self):
        return len(self.files)

    def __get_files(self, folder):
        files = os.listdir(folder)
        files = [os.path.join(folder, file) for file in files if file.endswith('.mat')]
        files = sorted(files)
        return files
    
    def __get_random_samples(self, x):
        if self.random_cut:
            random_start = torch.randint(0, x.size(-1) - self.samples, (1,))
        else:
            random_start = 42 * 500
        
        return x[:, random_start:random_start+self.samples]
        
    
    def __getitem__(self, idx):
        fname = self.files[idx]
        data = loadmat(fname)
        mecg = self.transform(data['out']['mecg'][0][0].astype(float))  # maternal ECG
        fecg = self.transform(data['out']['fecg'][0][0].astype(float))  # fetal ECG    
        sum_ = self.diffuser(mecg + fecg, self.snr_db)  # might vary the SNR
        fecg = self.stft.stft_only(fecg)
        fecg = fecg[:, :-1, :]  # drop last freq bin  
        sum_ = self.stft.stft_only(sum_)
        sum_ = sum_[:, :-1, :]  # drop last freq bin
        
        sum_ = sum_ / 10.0  # scale as close to [-1, 1]
        fecg = fecg / 10.0
        
        return sum_, fecg
    
    

def main():
    from fourier import STFT

    dset_path = 'data/ecg'
    dset = SignalDataset(dset_path)
        
    for i in range(len(dset)):
        data = dset[i]
        mix, fecg = data
        text = f'mix: {mix.shape}, {mix.dtype},'
        text += f'real: [{mix.real.min().item():.2f}, {mix.real.max().item():.2f}],'
        text += f'imag: [{mix.imag.min().item():.2f}, {mix.imag.max().item():.2f}]'
        print( text)
        
        text = f'fecg: {fecg.shape}, {fecg.dtype},'
        text += f'real: [{fecg.real.min().item():.2f}, {fecg.real.max().item():.2f}],'
        text += f'imag: [{fecg.imag.min().item():.2f}, {fecg.imag.max().item():.2f}]'
        print(text)
        
        if i == 5:
            break

    
    
if __name__ == '__main__':
    main()
