import os
import torch
import torchaudio
from scipy.io import loadmat
from tqdm import tqdm

from diffuser import Diffuser


class SignalDataset(torch.utils.data.Dataset):
    def __init__(self, folder, seconds_per_sample=3.83, snr_db=[5, 20], stft=None, num_channels=4, debug=False):
        self.files = self.__get_files(folder)
        resample_fn = torchaudio.transforms.Resample(orig_freq=2500, new_freq=500)
        self.samples = int(seconds_per_sample * 500)
        self.diffuser = Diffuser(sample_rate=500, snr_db=snr_db)
        self.transform = lambda x: self.__get_random_samples(resample_fn(torch.tensor(x, dtype=torch.float32).unsqueeze(0)))
        self.snr_db = snr_db
        self.random_cut = not 'test' in folder
        self.stft = stft
        self.num_channels = num_channels
        self.debug = debug
        
    def __len__(self):
        return self.num_channels * len(self.files)

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
        fname = self.files[idx // 4] # 4 channels per file
        selected_channel = idx % 4
        
        data = loadmat(fname)
        mecg = data['out']['mecg'][0][0].astype(float) # maternal ECG with num_channels
        fecg = data['out']['fecg'][0][0].astype(float) # fetal ECG with num_channels
        
        if self.debug:
            print(f'Shapes before transform: M:{mecg.shape}, f:{fecg.shape}')

        mecg = self.transform(mecg[selected_channel])  # transform also adds back the channel dimension
        fecg = self.transform(fecg[selected_channel])
        
        if self.debug:
            print(f'Shapes after transform: M:{mecg.shape}, f:{fecg.shape}')
        
        sum_ = self.diffuser(mecg + fecg)  # might vary the SNR
        fecg = self.stft.stft(fecg)
        fecg = fecg[:, :-1, :]  # drop last freq bin
        sum_ = self.stft.stft(sum_)
        sum_ = sum_[:, :-1, :]  # drop last freq bin
        
        if torch.isnan(sum_).any() or torch.isnan(fecg).any():
            print('Nan values found')
            print(fname)
            print(sum_.min(), sum_.max())
            print(fecg.min(), fecg.max())
            raise ValueError('Nan values found')
        
        return sum_, fecg
    
    

def main():
    from fourier import STFT

    dset_path = 'data/ecg'
    stft = STFT()
    dset = SignalDataset(dset_path, stft=stft, num_channels=4, debug=True)
    
    for i in range(10):
        x, y = dset[i]
    
    
if __name__ == '__main__':
    main()
