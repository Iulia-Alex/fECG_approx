import click
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from fourier import STFT
from logger import Logger
from loss import ComplexMSE
from network import ComplexUNet
from dataset import SignalDataset


class Trainer:
    def __init__(self, best_model_fname, logfile):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_model_fname = best_model_fname
        self.logger = Logger(logfile)
        
    def one_epoch(self, model, loader, train=True):
        model.train() if train else model.eval()
        total_loss = 0
        for x, y in tqdm(loader, desc='Batch', leave=False):
            x, y = x.to(self.device), y.to(self.device)
            y_pred = model(x)
            loss = self.loss_fn(y_pred, y)
            total_loss += loss.item()
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return total_loss / len(loader)
                
          

    def train(self, model, loaders, epochs):
        self.logger.max_epochs = epochs
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.loss_fn = ComplexMSE()
        model = model.to(self.device)
        for epoch in tqdm(range(epochs), desc='Epoch', leave=True):
            train_loss = self.one_epoch(model, loaders['train'])
            test_loss = self.one_epoch(model, loaders['test'], train=False)
            self.logger.log(train_loss, test_loss, epoch, model, self.best_model_fname)
        self.logger.draw_history()


def get_loaders(dataset, batch_size, seed):
    random_gen = torch.Generator().manual_seed(seed)
    train_set, test_set = torch.utils.data.random_split(dataset, [0.9, 0.1], generator=random_gen)
    test_set.random_cut = False
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=10)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=10)
    return {'train': train_loader, 'test': test_loader}


    
@click.command()
@click.option('-e', '--epochs', default=5, help='Number of epochs to train the model')
@click.option('-d', '--data', default='data/ecg', help='Path to the dataset')
@click.option('-b', '--batch_size', default=16, help='Batch size for training')
@click.option('-s', '--snr', default=20, help='Signal to noise ratio for the dataset')
@click.option('-o', '--output', default='models/best.pth', help='Path to save the model')
@click.option('--seed', default=42, help='Random seed')
@click.option('--logfile', default='results/log.txt', help='Path to save the log file')
def main(epochs, data, batch_size, snr, output, seed, logfile):
    model = ComplexUNet(128 * 128)
    loaders = get_loaders(SignalDataset(data, snr_db=snr), batch_size, seed)
    
    trainer = Trainer(output, logfile)
    trainer.train(model, loaders, epochs)
    
    
    
if __name__ == '__main__':
    main()