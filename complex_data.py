import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision
import os
import torch
import numpy as np
from complex_prepare_data import create_spectrogram

class SpectrogramDataset(Dataset):
    def __init__(self, root_path):
        self.files = [os.path.join(root_path, file) for file in os.listdir(root_path) if file.endswith('.mat')]
        self.files = sorted(self.files)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # torchvision.ops.Permute([1, 2, 0]), 
            transforms.Resize((128, 128))
        ])
        
    def __getitem__(self, idx):
        current_mat = self.files[idx]
        ### procesare, scoate spect
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "ecg")
        spect_dir = os.path.join(current_dir, "data/processed/spectrogram")
        input_spec, output_spec, input_phase, output_phase, _ = create_spectrogram(data_path, idx)
        input_spec = input_spec * np.exp(1j * input_phase)
        output_spec = output_spec * np.exp(1j * output_phase)
        input_spec.view(dtype=np.complex64)
        output_spec.view(dtype=np.complex64)
        input_spec = np.moveaxis(input_spec, 0, -1)
        input_spec = self.transform(input_spec)
        output_spec = np.moveaxis(output_spec, 0, -1)
        output_spec = self.transform(output_spec)
        return input_spec, output_spec

    def __len__(self):
        return len(self.files)
    
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "ecg")

    dset = SpectrogramDataset(data_path)
    
    for i in range(1,2):
        x, y = dset[i]
        print(x.shape, y.shape)