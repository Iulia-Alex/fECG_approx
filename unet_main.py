# import torch
# from torch import nn
# import torchvision
# import torchvision.transforms as transforms
# import numpy as np
# from PIL import Image
# import os
# from unet_utils import DoubleConvLayer, DownSampleLayer, UpSampleLayer
# from prepare_data import *

# import matplotlib.pyplot as plt

# class UNet(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.downconv1 = DownSampleLayer(in_channels, 64)
#         self.downconv2 = DownSampleLayer(64, 128)
#         self.downconv3 = DownSampleLayer(128, 256)
#         self.downconv4 = DownSampleLayer(256, 512)

#         self.bottleneck = DoubleConvLayer(512, 1024)

#         self.upconv1 = UpSampleLayer(1024, 512)
#         self.upconv2 = UpSampleLayer(512, 256)
#         self.upconv3 = UpSampleLayer(256, 128)
#         self.upconv4 = UpSampleLayer(128, 64)

#         self.out = nn.Conv2d(in_channels=64, out_channels=in_channels, kernel_size = 1)

#     def forward(self, x):
#         down1, p1 = self.downconv1(x)
#         down2, p2 = self.downconv2(p1)
#         down3, p3 = self.downconv3(p2)
#         down4, p4 = self.downconv4(p3)

#         bottle = self.bottleneck(p4)

#         up1 = self.upconv1(bottle, down4)
#         up2 = self.upconv2(up1, down3)
#         up3 = self.upconv3(up2, down2)
#         up4 = self.upconv4(up3, down1)

#         out = self.out(up4)

#         return out

# def load_image(image_path, input_size):
#     transform = transforms.Compose([
#         transforms.Resize(input_size),
#         transforms.ToTensor(),
#     ])
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image)
#     return image

# if __name__ == "__main__":
#     model = UNet(4)
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     model_path = os.path.join(current_dir, "models", "unet.pth")
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     image_path = os.path.join(current_dir, "C:/Users/ovidiu/Downloads/")
#     input_size = (256, 256)
#     (input_image, target_image) = create_spectrogram(image_path, 2917, model_path)
#     print(input_image.shape)
#     input_image = torch.tensor(input_image) #resize((256, 256)).unsqueeze(0)
#     input_image = transforms.functional.resize(input_image, (256, 256)).unsqueeze(0)
#     input_image=input_image.float()
#     print(input_image.shape, input_image.dtype)
#     output = model(input_image)
#     save_path = os.path.join(current_dir, "test.jpg")
#     # torchvision.utils.save_image(output.squeeze(0), save_path)

#     output = transforms.functional.resize(output, target_image.shape[-2:])
#     print(f"new shape", output.shape)
#     output = output.squeeze(0).detach().numpy()
#     print('Difference:', np.mean(np.abs(target_image-output)))


#     r, g, b, a = output
#     rt, gt, bt, at = target_image

#     fig, ax = plt.subplots(3,4)

#     ax[0][0].imshow(r)
#     ax[0][1].imshow(g)
#     ax[0][2].imshow(b)
#     ax[0][3].imshow(a)

#     ax[1][0].imshow(rt)
#     ax[1][1].imshow(gt)
#     ax[1][2].imshow(bt)
#     ax[1][3].imshow(at)

#     ax[2][0].imshow(rt -r)
#     ax[2][1].imshow(gt-g)
#     ax[2][2].imshow(bt-b)
#     ax[2][3].imshow(at-a)

#     for i in range(3):
#         for j in range(4):
#             ax[i][j].axis('off')

#     plt.show()

#     output_sc = scale_minmax(output, -80, 0)
    # output_v = librosa.db_to_amplitude(output_sc, ref=1.0)
#     output_ecg = librosa.istft(output_v, hop_length=128, win_length=256, n_fft=512) # sau .T?

#     fig, ax = plt.subplots(2,1)
#     ax[0].plot(output_ecg[0, :])
#     data = sio.loadmat(current_dir + "/fecgsyn" + 'test' + '.mat')
#     out_data = data['out']
#     fecg = out_data['fecg'][0, 0]
#     fecg = fecg.astype(float)
#     print("shape",fecg.shape)
#     fecg=librosa.resample(y=fecg, orig_sr=2500, target_sr=500)
#     print("new shape", fecg.shape)
#     ax[1].plot(fecg[0, :])
#     plt.show()

import torch
from torch import nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.signal as signal
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

def unet_main(model, image_path, index, phase = 'Fetal', channel = 0):
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

    # Create spectrogram
    input_size = (128, 128)
    input_image, target_image, phase, ref = create_spectrogram(image_path, index, True, phase)
    
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
    
    # # Visualization
    # r, g, b, a = output
    # rt, gt, bt, at = target_image

    # fig, ax = plt.subplots(3, 4, figsize=(12, 8))

    # # Model Output spect on channel 1, 2, 3, 4
    # ax[0][0].imshow(r)
    # ax[0][0].set_title('Model Output spectrogram - Channel 1')
    # ax[0][1].imshow(g)
    # ax[0][1].set_title('Model Output spectrogram - Channel 2')
    # ax[0][2].imshow(b)
    # ax[0][2].set_title('Model Output spectrogram - Channel 3')
    # ax[0][3].imshow(a)
    # ax[0][3].set_title('Model Output spectrogram - Channel 4')

    # # Targeted spect on channel 1, 2, 3, 4
    # ax[1][0].imshow(rt)
    # ax[1][0].set_title('Targeted spectrogram - Channel 1')
    # ax[1][1].imshow(gt)
    # ax[1][1].set_title('Targeted spectrogram - Channel 2')
    # ax[1][2].imshow(bt)
    # ax[1][2].set_title('Targeted spectrogram - Channel 3')
    # ax[1][3].imshow(at)
    # ax[1][3].set_title('Targeted spectrogram - Channel 4')

    # # The difference computed between targeted spect and output spect on channel 1, 2, 3, 4
    # ax[2][0].imshow(rt - r)
    # ax[2][0].set_title('Difference - Channel 1')
    # ax[2][1].imshow(gt - g)
    # ax[2][1].set_title('Difference - Channel 2')
    # ax[2][2].imshow(bt - b)
    # ax[2][2].set_title('Difference - Channel 3')
    # ax[2][3].imshow(at - a)
    # ax[2][3].set_title('Difference - Channel 4')

    # # Turn off axis for all subplots
    # for i in range(3):
    #     for j in range(4):
    #         ax[i][j].axis('off')

    # plt.tight_layout()
    # plt.show()

    # Reconstruct the ECG signal from the spectrogram
    # output_sc = scale_minmax(output * 255, -80, 0)
    output_sc = output * 80 - 80
    output_ecg = istft(output_sc, 256, 15, 100, phase, 2000, ref)

    # Plot the reconstructed ECG signal
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(output_ecg[channel, :])
    ax[0].set_title("Reconstructed ECG Signal on channel " + str(channel))
    ax[0].set_xlabel('Samples')
    ax[0].set_ylabel('Amplitude')
    
    # Load the original signal for comparison
    if index > 0 and index < 10:
        str_i = '0' + str(index)
    else:
        str_i = str(index)
    data = sio.loadmat(os.path.join(image_path, "fecgsyn" + str_i + '.mat'))
    out_data = data['out']
    fecg = out_data['fecg'][0, 0].astype(float)
    fecg = librosa.resample(y=fecg, orig_sr=2500, target_sr=500)
    ax[1].plot(fecg[channel, 5000:7000])
    ax[1].set_title("Original ECG Signal, channel" + str(channel))
    ax[1].set_xlabel('Samples')
    ax[1].set_ylabel('Amplitude')
    plt.show()

    rpeaks = sleepecg.detect_heartbeats(output_ecg[channel,:], fs=500)
    peak_values = output_ecg[channel, rpeaks]
    plt.figure()
    plt.plot(output_ecg[channel, :], label='Predicted signal')
    plt.plot(rpeaks, peak_values, 'ro', label='Detected peaks')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()

    detected_peaks_in_seconds = (np.asarray(rpeaks) + 5000) / 500
    print('First Detector:')
    HR = 60 / (np.diff(detected_peaks_in_seconds))
    # print(f'Computed heart rates', HR)
    filtered_heart_rates = remove_outliers_iqr(HR)
    # print(f"Filtered heart rates:", filtered_heart_rates)
    print(f'Estimated HR V1:', np.mean(filtered_heart_rates))

    rpeaks = wfdb.processing.xqrs_detect(output_ecg[channel,:], fs=500, verbose=False)
    peak_values = output_ecg[channel, rpeaks]
    plt.figure()
    plt.plot(output_ecg[channel, :], label='Predicted signal')
    plt.plot(rpeaks, peak_values, 'ro', label='Detected peaks')
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

    detected_peaks_in_seconds = (np.asarray(rpeaks) + 5000) / 500
    print(f'Detected Peaks in Seconds:', detected_peaks_in_seconds)
    HR = 60 / (np.diff(detected_peaks_in_seconds))
    print('Second Detector:')
    print(f'Computed heart rates', HR)
    filtered_heart_rates1 = remove_outliers_iqr(HR)
    print(f"Filtered heart rates:", filtered_heart_rates1)
    print(f'Estimated HR V2:', np.mean(filtered_heart_rates1))
    print(f'Real HR:', out_data['param'][0,0]['fhr'][0,0].astype(float))
    return np.mean(filtered_heart_rates), np.mean(filtered_heart_rates1), out_data['param'][0,0]['fhr'][0,0].astype(float)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "C:/Users/ovidiu/Downloads/fecgsyn_test/")
    # for i in range(1, 101):
    unet_main('AttentionUnet', image_path, 1, 'Fetal')