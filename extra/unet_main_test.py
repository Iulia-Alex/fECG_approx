import torch
from torch import nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.signal
from scipy.io import savemat
import librosa
from ecgdetectors import Detectors
from unet_utils import *
from prepare_data import *
from qrs_detection import *
import wfdb.processing
import neurokit2
import sleepecg

class UNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.downconv1 = DownSampleLayer(in_channels, 64)
        self.downconv2 = DownSampleLayer(64, 128)
        self.downconv3 = DownSampleLayer(128, 256)
        self.downconv4 = DownSampleLayer(256, 512)

        self.bottleneck = DoubleConvLayer(512, 1024)

        self.upconv1 = UpSampleLayer(1024, 512)
        self.upconv2 = UpSampleLayer(512, 256)
        self.upconv3 = UpSampleLayer(256, 128)
        self.upconv4 = UpSampleLayer(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        down1, p1 = self.downconv1(x)
        down2, p2 = self.downconv2(p1)
        down3, p3 = self.downconv3(p2)
        down4, p4 = self.downconv4(p3)

        bottle = self.bottleneck(p4)

        up1 = self.upconv1(bottle, down4)
        up2 = self.upconv2(up1, down3)
        up3 = self.upconv3(up2, down2)
        up4 = self.upconv4(up3, down1)

        out = self.out(up4)
        # out = self.sigmoid(out)

        return out

def remove_outliers_iqr(data):
    Q1 = np.percentile(data, 15)
    Q3 = np.percentile(data, 85)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    
    return np.array(filtered_data)

def cola_reconstruct(frames, hop_size):
    # Initialize the output signal array with zeros
    num_frames, frame_size = frames.shape
    output_length = hop_size * (num_frames - 1) + frame_size
    output_signal = np.zeros(output_length)

    # Overlap-add the frames into the output signal
    for i in range(num_frames):
        start = i * hop_size
        output_signal[start:start + frame_size] += frames[i]

    return output_signal

def unet_main(model):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "C:/Users/ovidiu/Downloads/fecgsyn_test/")
    # Load the trained model
    if model == 'UNet':
        model = UNet(4)
        # model_path = os.path.join("D:\Iulia\ETTI\Licenta\Python UNet\models", "unet_5.pth")
        model_path = os.path.join("D:/Iulia/ETTI/Licenta/Licenta/unet/models", "unet_10.pth")
    elif model == 'AttentionUnet':
        model = AttU_Net()
        model_path = os.path.join("D:/Iulia/ETTI/Licenta/Licenta/unet/models", "unet_14.pth") #4
    else:
        raise ValueError("No model loaded")
        
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    for index in range(1, 101):

        if index > 0 and index < 10:
            str_i = '0' + str(index)
        else:
            str_i = str(index)
        data = sio.loadmat(os.path.join(image_path, "fecgsyn" + str_i + '.mat'))
        out_data = data['out']
        fecg = out_data['fecg'][0, 0].astype(float)
        mecg = out_data['mecg'][0, 0].astype(float)

        mecg = librosa.resample(mecg, orig_sr=2500, target_sr=500)
        fecg = librosa.resample(fecg, orig_sr=2500, target_sr=500)

        sum_ = mecg + fecg

        sum_noise, noise = add_noise(sum_, 4, 500, sum_.shape[1], 5)
        signal = sum_noise

        signal_length = 500 * 60  # lungimea semnalului original
        frame_size = 2000  # mărimea fiecărui segment (fereastră)
        hop_size = 1500  # cât de mult suprapui segmentele

        full_length_ecg_0 = np.zeros([4, signal_length])
        full_length_ecg_1 = np.zeros([4, signal_length])

        window = scipy.signal.windows.hann(frame_size, sym=False)

        num_frames = (signal_length - frame_size) // hop_size + 1
        frames = np.zeros((4, num_frames, frame_size))
        fecg_frames = np.zeros((4, num_frames, frame_size))

        for i in range(num_frames):
            start = i * hop_size
            frames[:,i] = signal[:,start:start + frame_size] * window
            fecg_frames[:,i] = fecg[:,start:start + frame_size]

        for i in range(num_frames):
            if (scipy.signal.check_COLA(window, frame_size, hop_size)):
                # Create spectrogram
                input_size = (128, 128)
                input_image, target_image, phase, ref = create_spectrogram1(fecg_frames[:,i], frames[:,i], 'Fetal')

                # Preprocess input image
                input_image = torch.tensor(input_image).float()
                input_image = transforms.functional.resize(input_image, input_size).unsqueeze(0)
                
                # Perform inference
                with torch.no_grad():
                    output = model(input_image)
                
                # Resize the output to match the target image's shape
                output = transforms.functional.resize(output, target_image.shape[-2:])
                
                # Convert the output tensor to numpy array
                output = output.squeeze(0).detach().numpy()

                # Reconstruct the ECG signal from the spectrogram
                output_sc = output * 80 - 80
                output_ecg = istft(output_sc, 256, 15, 100, phase, 2000, ref)
                # full_length_ecg_0[:, i*frame_size:(i+1)*frame_size] = output_ecg
                start = i * hop_size
                full_length_ecg_0[:,start:start + frame_size] += output_ecg

                # ########################

                # Create spectrogram
                input_size = (128, 128)
                input_image, target_image, phase, ref = create_spectrogram1(fecg_frames[:,i], frames[:,i], 'Mixture')
                
                # Preprocess input image
                input_image = torch.tensor(input_image).float()
                input_image = transforms.functional.resize(input_image, input_size).unsqueeze(0)
                
                # Perform inference
                with torch.no_grad():
                    output = model(input_image)
                
                # Resize the output to match the target image's shape
                output = transforms.functional.resize(output, target_image.shape[-2:])
                
                # Convert the output tensor to numpy array
                output = output.squeeze(0).detach().numpy()

                # Reconstruct the ECG signal from the spectrogram
                output_sc = output * 80 - 80
                output_ecg = istft(output_sc, 256, 15, 100, phase, 2000, ref)
                full_length_ecg_1[:,start:start + frame_size] += output_ecg

        d = {'Reconstruction_0': full_length_ecg_0, 'Reconstruction_1': full_length_ecg_1, 'Original_fECG': fecg, 'Noisy_input': sum_noise}
        # savemat("result" + str(index) + ".mat", d)
        plt.figure(), plt.plot(mecg[0, :]), plt.show()
        plt.figure(), plt.plot(full_length_ecg_1[0,:], '-r'), plt.plot(fecg[0, :]), plt.show()
        print("Current file: fecgsyn" + str(index))
    
if __name__ == "__main__":
    complex_data = unet_main('AttentionUnet')
