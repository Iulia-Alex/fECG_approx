import torch
import torchaudio
import matplotlib.pyplot as plt
from scipy.io import loadmat


from iulia_old.prepare_data import add_noise
from fourier import STFT
from network import ComplexUNet
from dataset import SignalDataset
from train import get_loaders


def get_model(path, sizes=(128, 128)):
    model = ComplexUNet(sizes[0] * sizes[1])
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def predict(model, x, stft):
    x = torch.tensor(x, dtype=torch.float32)
    x = stft.stft(x)
    x = x.unsqueeze(0)
    with torch.no_grad():
        y = model(x)
    y = stft.istft(y.squeeze())
    return y.numpy()


### TODO: apply changes (fourier and dataset) to this script
if __name__ == '__main__':
    model_path = 'models/best.pth'
    signal_path = 'data/ecg/fecgsyn3008.mat'
    snr_db = 20

    loaders = get_loaders(SignalDataset('data/ecg', snr_db=snr_db), 32, 42)
    loader = loaders['test']
    
    model = get_model(model_path).to('cuda')
    stft = STFT()
    
    mix, fecg = next(iter(loader))
    
    lengths = [x.size(-1) for x in mix]
    mix_specs = stft.stft_batched(mix)
    with torch.no_grad():
        pred = model(mix_specs.to('cuda'))
    pred = stft.istft_batched(pred.to('cpu'), lengths)
    
    print('mix', mix.shape, 'fecg', fecg.shape, 'pred', pred.shape)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(mix[0][0].numpy())
    plt.title('Mixed signal')
    plt.subplot(1, 3, 2)
    plt.plot(fecg[0][0].numpy())
    plt.title('Fetal ECG')
    plt.subplot(1, 3, 3)
    plt.plot(pred[0][0].numpy())
    plt.title('Predicted signal')
    plt.savefig('results.png')
    
       











