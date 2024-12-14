import click
import torch
from tqdm import tqdm


from train import Trainer
from network import create_model
from dataset import SignalDataset
from fourier import STFT




def train(train_files, excluded_files):
    
    dset_path = 'data/abdominal_fecg'
    stft = STFT()
    batch_size = 16
    workers = 8
    
    dset_train = SignalDataset(dset_path, stft=stft, num_channels=4, leave_one_out=excluded_files)
    dset_val = SignalDataset(dset_path, stft=stft, num_channels=4, leave_one_out=train_files)
    print(f"Training with {dset_train.files} and validating with {dset_val.files}")
    
    train_loader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = torch.utils.data.DataLoader(dset_val, batch_size=batch_size, shuffle=False, num_workers=workers)
    loaders = {'train': train_loader, 'test': val_loader, 'stft': stft}

    model = create_model('models/currently_best_372.pth')
    save_model = f'models/abdominal_finetune/out_{excluded_files[0].split("_")[0]}.pth'
    logfile = f'logs/abdominal_finetune/out_{excluded_files[0].split("_")[0]}.txt'
    
    trainer = Trainer(save_model, logfile)
    trainer.train(model, loaders, epochs=100, lr=5e-4)



if __name__ == '__main__':
    
    files = ['01', '04', '07', '08', '10']
    files = [f'r{f}_test.mat' for f in files]
    
    for file in files:
        file_to_train = [f for f in files if f != file]
        files_to_exclude = [f for f in files if f == file]
        
        train(train_files=file_to_train, excluded_files=files_to_exclude)
    