import torch
import colorednoise as cn


class Diffuser():
    def __init__(self, num_channels, sample_rate, beta=1):
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.beta = beta

    def mixtgauss(self, N, p, sigma0, sigma1):
        q = torch.randn(N)
        u = q < p
        x = (sigma0 * torch.logical_not(u) + sigma1 * u) * torch.randn(N)
        return x

    def add_noise(self, ecg, nb_samples, snr_db):
        ecg_plus_noise = ecg.clone().detach().T
        noise_arr = torch.empty(ecg_plus_noise.shape)
        for i in range(self.num_channels):
            
            pink_noise = torch.tensor(cn.powerlaw_psd_gaussian(self.beta, nb_samples))
            white_noise = torch.randn(nb_samples)
            mixture_gaussian = self.mixtgauss(nb_samples, 0.1, 1, 10)
            
            pink_fft = torch.fft.fft(pink_noise)
            white_fft = torch.fft.fft(white_noise)
            mixture_fft = torch.fft.fft(mixture_gaussian)
            
            pink_range = int(torch.rand(1) * 4 + 9)
            num_samples_pink = pink_range * nb_samples // self.sample_rate
            white_range = int(torch.rand(1) * 30 + 60)
            num_samples_white = white_range * nb_samples // self.sample_rate
            
            pink_fft =  torch.abs(pink_fft) / max(torch.abs(pink_fft[:num_samples_pink]))
            white_fft = torch.abs(white_fft) / max(torch.abs(white_fft[num_samples_pink:num_samples_pink+num_samples_white]))
            mixture_fft = torch.abs(mixture_fft) / max(torch.abs(mixture_fft[num_samples_pink+num_samples_white:]))
            combined_fft = torch.zeros_like(pink_fft)
            combined_fft += 2.0 * pink_fft + 0.2 * white_fft + 0.15 * mixture_fft
            noise_timedomain = torch.real(torch.fft.ifft(combined_fft))
            
            rms_signal = torch.sqrt(torch.mean(torch.square(ecg)))
            rms_noise_exp = 10.0 ** (torch.log10(rms_signal) - snr_db / 20.0)
            normalized_noise = (noise_timedomain - torch.mean(noise_timedomain)) / torch.std(noise_timedomain)
            new_noise = rms_noise_exp * normalized_noise
            ecg_plus_noise[:, i] += new_noise
            noise_arr[:, i] = new_noise
        return (ecg_plus_noise.T, noise_arr.T)
    
    
    def __call__(self, ecg, snr_db=10):
        nb_samples = ecg.size(-1)
        ecg_plus_noise, noise_arr = self.add_noise(ecg, nb_samples, snr_db)
        return ecg_plus_noise


def main():
    from iulia_old.prepare_data import add_noise as add_noise_old
    
    num_channels = 3
    sample_rate = 500
    diffuser = Diffuser(num_channels, sample_rate)
    
    ecg = torch.randn(num_channels, 5000)
    nb_samples = ecg.size(-1)
    snr_db = 10
    
    ecg_plus_noise, noise_arr = diffuser.add_noise(ecg, nb_samples, snr_db)
    ecg_plus_noise_old, noise_arr_old = add_noise_old(ecg.numpy(), num_channels, sample_rate, nb_samples, snr_db)
    
    print('ecg_plus_noise', ecg_plus_noise.shape, ecg_plus_noise_old.shape)
    print('noise_arr', noise_arr.shape, noise_arr_old.shape)
    
    print('ecg_plus_noise', ecg_plus_noise.max().item(), ecg_plus_noise_old.max())
    print('ecg_plus_noise', ecg_plus_noise.min().item(), ecg_plus_noise_old.min())
    print('noise_arr', noise_arr.max().item(), noise_arr_old.max())
    print('noise_arr', noise_arr.min().item(), noise_arr_old.min())


if __name__ == '__main__':
    main()
    
    
    