import os
import torch
import torchaudio
from scipy.io import loadmat
from tqdm import tqdm

from diffuser import Diffuser


def load_mat_ours(path):
    data = loadmat(path)
    mecg = data['out']['mecg'][0][0].astype(float)
    fecg = data['out']['fecg'][0][0].astype(float)
    fecg = torch.tensor(fecg, dtype=torch.float32)
    mecg = torch.tensor(mecg, dtype=torch.float32)
    return mecg, fecg

def load_mat_abdominal(path):
    data = loadmat(path)
    data = data['out']
    sum_ = data['fecg'][0][0].astype(float)
    fecg = data['real'][0][0].astype(float)
    sum_, fecg = sum_.T, fecg.T
    sum_ = torch.tensor(sum_, dtype=torch.float32)
    fecg = torch.tensor(fecg, dtype=torch.float32)
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
    # EWAM ?
    orig_sr = 2500
    new_sr = 500
    resampler = torchaudio.transforms.Resample(orig_sr, new_sr)
    
    data = loadmat(path)
    signal = data['s'].astype(float)
    signal = torch.tensor(signal, dtype=torch.float32)
    signal = signal.view(1, -1)
    signal = resampler(signal)
    return signal[:, :size]


class SignalDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 folder, 
                 seconds_per_sample=3.83, 
                 snr_db=[5, 20], 
                 stft=None, 
                 num_channels=4,
                 debug=False,
                 leave_one_out=None,
                 overiding=1
        ):
        if type(leave_one_out) == str:
                self.leave_one_out = [leave_one_out]
        else:
            self.leave_one_out = leave_one_out

        self.files = self.__get_files(folder)
        self.sr = 500  # the sample rate of interest for our network
        self.samples = int(seconds_per_sample * self.sr)
        self.diffuser = Diffuser(sample_rate=self.sr, snr_db=snr_db)
        # self.transform = lambda x: self.__get_random_samples(resample_fn(torch.tensor(x, dtype=torch.float32).unsqueeze(0)))
        self.snr_db = snr_db
        self.random_cut = not 'test' in folder
        self.stft = stft
        self.num_channels = num_channels
        self.debug = debug
        # determine the dataset type
        if 'abdominal' in folder:
            self.dset_type = 'abdominal'
            self.init_sr = 100
            self.overiding = 50
        else:  # our dataset
            self.dset_type = 'ours'
            self.init_sr = 2500
            self.overiding = overiding
        self.resample_fn = torchaudio.transforms.Resample(self.init_sr, self.sr)

    def __len__(self):
        # overiding the length of the dataset to select same file multiple times
        return self.overiding * self.num_channels * len(self.files)

    def __get_files(self, folder):
        files = os.listdir(folder)
        files = [os.path.join(folder, file) for file in files if file.endswith('.mat')]
        files = sorted(files)
        if self.leave_one_out:
            self.leave_one_out = [os.path.join(folder, file) for file in self.leave_one_out]
            files = [f for f in files if f not in self.leave_one_out]
        return files
    
    def __get_random_samples(self, x):
        if self.random_cut:
            random_start = torch.randint(0, x.size(-1) - self.samples, (1,))
        else:
            random_start = 42 * self.sr # 42 seconds, for good luck
        
        return x[:, random_start:random_start+self.samples]


    def __process_file(self, x, channel):
        x = x[channel].unsqueeze(0)
        x = self.resample_fn(x)
        x = self.__get_random_samples(x)
        return x


    def __getitem__(self, idx):
        fname = self.files[idx // (self.num_channels * self.overiding)] # determine the file
        selected_channel = (idx // self.overiding) % self.num_channels  # determine the channel
        
        if self.dset_type == 'ours':
            mecg, fecg = load_mat_ours(fname)
        elif self.dset_type == 'abdominal':
            mecg, fecg = load_mat_abdominal(fname)
        
        mecg = self.__process_file(mecg, selected_channel)
        fecg = self.__process_file(fecg, selected_channel)
        
        if self.dset_type == 'ours':
            sum_ = self.diffuser(mecg + fecg)
        elif self.dset_type == 'abdominal':
            sum_ = mecg
    
        if self.debug:
            print(f'File: {fname}')
            print(f'Sum: {sum_.shape}, Fecg: {fecg.shape}')
            
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
    
    

def test_parser():
    from fourier import STFT
    stft = STFT()

    # dset_path = 'data/ecg'
    # dset = SignalDataset(dset_path, stft=stft, num_channels=4, debug=False)
    # for (x, y) in tqdm(dset, total=len(dset)):
    #     if list(x.shape) != [1, 128, 128] or list(y.shape) != [1, 128, 128]:
    #         print(x.shape, y.shape)
    #         raise ValueError('Invalid shape')
    
    # print("[INFO] First test passed (our dset) !")
    
    dset_path = 'data/abdominal_fecg'
    dset = SignalDataset(dset_path, stft=stft, num_channels=4, debug=False)
    for (x, y) in tqdm(dset, total=len(dset)):
        if list(x.shape) != [1, 128, 128] or list(y.shape) != [1, 128, 128]:
            print(x.shape, y.shape)
            raise ValueError('Invalid shape')
    print("[INFO] Second test passed (abdominal dset) !")
    print('[DONE] All tests passed!')
    
    
def test_one_out():
    from fourier import STFT
    
    dset_path = 'data/abdominal_fecg'
    stft = STFT()
    
    dset = SignalDataset(dset_path, stft=stft, num_channels=4, debug=False)
    files = dset.files
    print(f"Files in the dataset: {len(files)}")
    for file in files:
        print(file)
    
    files_to_exclude = ['01']
    files_to_exclude = [f'r{f}_test.mat' for f in files_to_exclude]
    dset_out = SignalDataset(dset_path, stft=stft, num_channels=4, debug=False, leave_one_out=files_to_exclude)
    files_out = dset_out.files
    print(f"Files in the dataset without 01: {len(files_out)}")
    for file in files_out:
        print(file)

    files_to_exclude = ['01', '04']
    files_to_exclude = [f'r{f}_test.mat' for f in files_to_exclude]
    dset_out2 = SignalDataset(dset_path, stft=stft, num_channels=4, debug=False, leave_one_out=files_to_exclude)
    files_out2 = dset_out2.files
    print(f"Files in the dataset without 01 and 04: {len(files_out2)}")
    for file in files_out2:
        print(file)
    
    
if __name__ == '__main__':
    test_parser()
    # test_one_out()