import os
import librosa
import numpy as np
import scipy.io as sio
from scipy.io import loadmat
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import colorednoise as cn

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def mixtgauss(N, p, sigma0, sigma1):
    '''
    gives a mixture of gaussian noise
    arguments:
    N: data length
    p: probability of peaks
    sigma0: standard deviation of background noise
    sigma1: standard deviation of impulse noise
    output: x: output noise
    '''
    q = np.random.randn(N)
    u = q < p
    x = (sigma0 * (1 - u) + sigma1 * u) * np.random.randn(N)
    return x

def add_noise(ecg, num_channels, sample_rate, nb_samples, snr_db):
    ecg_plus_noise = ecg.T
    noise_arr = np.empty(ecg_plus_noise.shape)
    for i in range(num_channels):
        beta = 1
        pink_noise = cn.powerlaw_psd_gaussian(beta, nb_samples)
        white_noise = np.random.randn(nb_samples)
        mixture_gaussian = mixtgauss(nb_samples, 0.1, 1, 10)

        pink_fft = np.fft.fft(pink_noise)
        white_fft = np.fft.fft(white_noise)
        mixture_fft = np.fft.fft(mixture_gaussian)

        pink_range = int(np.random.rand() * 4 + 9)
        num_samples_pink = pink_range * nb_samples // sample_rate
        white_range = int(np.random.rand() * 30 + 60)
        num_samples_white = white_range * nb_samples // sample_rate

        pink_fft =  np.abs(pink_fft) / max(np.abs(pink_fft[:num_samples_pink]))
        white_fft = np.abs(white_fft) / max(np.abs(white_fft[num_samples_pink:num_samples_pink+num_samples_white]))
        mixture_fft = np.abs(mixture_fft) / max(np.abs(mixture_fft[num_samples_pink+num_samples_white:]))
        combined_fft = np.zeros_like(pink_fft)
        combined_fft += 2*pink_fft + 0.2*white_fft + 0.15*mixture_fft
        # combined_fft = np.zeros(nb_samples, dtype=np.complex128)
        # combined_fft[:num_samples_pink] = 2 * pink_fft[:num_samples_pink]
        # combined_fft[num_samples_pink:num_samples_pink+num_samples_white] = 0.2 * white_fft[num_samples_pink:num_samples_pink+num_samples_white]
        # combined_fft[num_samples_pink+num_samples_white:] = 0.15 * mixture_fft[num_samples_pink+num_samples_white:] #2 pink 0.15 mixt
        noise_timedomain = np.fft.ifft(combined_fft)

        # snr_db = np.random.rand() * 10 + 5
        rms_signal = np.sqrt(np.mean(np.square(ecg)))
        rms_noise_exp = 10 ** (np.log10(rms_signal) - snr_db / 20)
        normalized_noise = (noise_timedomain - np.mean(noise_timedomain)) / np.std(noise_timedomain)
        new_noise = rms_noise_exp * normalized_noise
        ecg_plus_noise[:, i] += new_noise.astype(float)
        noise_arr[:, i] = new_noise

    return (ecg_plus_noise.T, noise_arr.T)

def load_and_resample(mat, original_sr, desired_sr):
    mecg = np.real(mat['out']['mecg'][0][0]).astype(np.float32)  # maternal ECG
    fecg = np.real(mat['out']['fecg'][0][0]).astype(np.float32)  # fetal ECG

    mecg = librosa.resample(mecg, orig_sr=original_sr, target_sr=desired_sr)
    fecg = librosa.resample(fecg, orig_sr=original_sr, target_sr=desired_sr)
    sum_ = mecg + fecg
    
    return mecg, fecg, sum_

def shorten(signal, sr, start):
    signal = signal[:, start:start+(4*sr)]
    return signal

def stft(signal, nfft, hoplen, win_len):
    spec = librosa.stft(signal, n_fft=nfft, hop_length=hoplen, win_length=win_len)
    ref = np.max(np.abs(spec))
    spec_db = librosa.amplitude_to_db(np.abs(spec), ref=ref)
    phase = np.angle(spec)
    return spec_db, phase, ref


def istft(spec_db, nfft, hoplen, win_len, phase=None, length=None, ref=1):
    spec = librosa.db_to_amplitude(spec_db, ref=ref)
    if phase is not None:
        spec = spec * np.exp(1j * phase)
    signal = librosa.istft(spec, n_fft=nfft, hop_length=hoplen, win_length=win_len, length=length)
    return signal

def create_spectrogram(load_path, index, test=False, phase = 'Fetal', random_start=0, snr=5):
    if index > 0 and index < 10:
        str_index = '0' + str(index)
    else:
        str_index=str(index)
    file = load_path + "/fecgsyn" + str_index + '.mat'
    mat = loadmat(file)
    
    old_sr = 2500
    sr = 500
    
    mecg, fecg, sum_ = load_and_resample(mat, old_sr, sr)
    if test:
        random_start = 5000
    # else:
    #     random_start = np.random.randint(0, 56) * sr
    mecg = shorten(mecg, sr, random_start)
    fecg = shorten(fecg, sr, random_start)
    sum_ = shorten(sum_, sr, random_start) #added for test??
    sum_noise, noise = add_noise(sum_, 4, 500, sum_.shape[1], snr)
    # plt.plot(sum_noise[0, :]), plt.title('mECG + fECG + noise'), plt.xlabel('Time [Samples]'), plt.ylabel('Amplitude')
    # plt.figure(), plt.subplot(2,1,1), plt.plot(sum_noise[0,:])
    # plt.subplot(2,1,2), plt.plot(mecg[0,:])
    nfft = 256
    win_len = 100
    hoplen = 15
    
    fecg_spec, fecg_phase, ref = stft(fecg, nfft=nfft, hoplen=hoplen, win_len=win_len)
    sum_noise_spec, sum_noise_phase, ref1 = stft(sum_noise, nfft=nfft, hoplen=hoplen, win_len=win_len)
    # mecg_spec, mecg_phase, ref2 = stft(mecg, nfft=nfft, hoplen=hoplen, win_len=win_len)

    # sum_noise_spec1 = scale_minmax(sum_noise_spec, 0, 255).astype(np.uint8) / 255
    # fecg_spec1 = scale_minmax(fecg_spec, 0, 255).astype(np.uint8) / 255

    sum_noise_spec1 = (sum_noise_spec + 80) / 80
    fecg_spec1 = (fecg_spec + 80) / 80

    # sum_noise_spec1 = sum_noise_spec / 80 + 1
    # fecg_spec1 = fecg_spec / 80 + 1
    # # plt.figure(), plt.imshow(fecg_spec1[0,:,:]), plt.title('fECG spectrogram'), plt.colorbar(), plt.xlabel('Time'), plt.ylabel('Frequency'), plt.show()
    # extracted_spec = scale_minmax(fecg_spec1, -80, 0).astype(np.float32) #fecg_spec
    # extracted_spec = (fecg_spec1 - 1) * 80
    # signal = istft(extracted_spec, 256, 15, 100, fecg_phase, 2000, ref1)
    
    # plt.figure(), plt.plot(signal[1, :]), plt.plot(fecg[1,:]*(signal[1,1]/fecg[1,1]),'r')
    # plt.figure(), plt.subplot(4,1,1), plt.plot(fecg_phase[1, :, 15], "r")
    # plt.subplot(4,1,2), plt.plot(sum_noise_phase[1,:, 15], 'b')
    # plt.subplot(4,1,3), plt.plot(mecg_phase[1,:,15])
    # plt.subplot(4,1,4), plt.plot(sum_noise_phase[1,:,15]-mecg_phase[1,:,15])

    # plt.figure(), plt.plot(sum_noise[0, :]), plt.title('Mixture signal'), plt.xlabel('Time [samples]'), plt.ylabel('Amplitude'), plt.show()
    if phase == 'Fetal':
        return sum_noise_spec1, fecg_spec1, fecg_phase, ref1
    
    return sum_noise_spec1, fecg_spec1, sum_noise_phase, ref1
    
# create_spectrogram(r"D:/Iulia/ETTI/Licenta/Python UNet/ecg", 10)
# create_spectrogram(r"C:/Users/ovidiu/Downloads/", 3005, True)

def create_spectrogram1(fecg, sum_noise, phase = 'Fetal', snr=5):
    nfft = 256
    win_len = 100
    hoplen = 15
    
    fecg_spec, fecg_phase, ref = stft(fecg, nfft=nfft, hoplen=hoplen, win_len=win_len)
    sum_noise_spec, sum_noise_phase, ref1 = stft(sum_noise, nfft=nfft, hoplen=hoplen, win_len=win_len)
    
    sum_noise_spec1 = (sum_noise_spec + 80) / 80
    fecg_spec1 = (fecg_spec + 80) / 80
    if phase == 'Fetal':
        return sum_noise_spec1, fecg_spec1, fecg_phase, ref1
    
    return sum_noise_spec1, fecg_spec1, sum_noise_phase, ref1