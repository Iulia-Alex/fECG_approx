import click
import torch
from tqdm import tqdm

from logger import Logger
from loss import ComposedLoss, SignalMSE, SignalMAE
from fourier import STFT
from metrics import PDR
from network import ComplexUNet
from dataset import SignalDataset


class Trainer:
    def __init__(self, best_model_fname, logfile, debug=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_model_fname = best_model_fname
        self.logger = Logger(logfile)
        self.debug = debug


    def one_epoch(self, model, loader, train=True):
        model.train() if train else model.eval()
        total_loss = 0
        total_metrics = {'prd':0.0}
        for x, y in tqdm(loader, leave=False, bar_format='Batch: {l_bar}{bar:10}{r_bar}{bar:-10b}'):
            x, y = x.to(self.device), y.to(self.device)
            
            if self.debug:
                tqdm.write(f'x: {x.shape}, y: {y.shape}')
            
            if train:
                y_pred = model(x)
            else:
                with torch.no_grad():
                    y_pred = model(x)
            loss = self.loss_fn(y, y_pred)
            
            if self.debug:
                tqdm.write(f'Loss: {loss.item()}')
            
            total_loss += loss.item()
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            metrics_dict = self.metrics(y, y_pred)
            total_metrics['prd'] += metrics_dict['prd']
                
        loss = total_loss / len(loader)
        total_metrics['prd'] /= len(loader)
        return loss, total_metrics


    def train(self, model, loaders, epochs):
        self.logger.max_epochs = epochs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.loss_fn = SignalMAE(loaders['stft'])
        self.metrics = PDR(loaders['stft'])
        model = model.to(self.device)
        for epoch in tqdm(range(epochs), leave=False, bar_format='Epoch: {l_bar}{bar:10}{r_bar}{bar:-10b}'):
            train_loss, metrics_train = self.one_epoch(model, loaders['train'])
            test_loss, metrics_test = self.one_epoch(model, loaders['test'], train=False)
            loss = {'train': train_loss, 'test': test_loss}
            metrics = {'train': metrics_train, 'test': metrics_test}
            self.logger.log(loss, metrics, epoch, model, self.best_model_fname)
        self.logger.draw_history()




    
@click.command()
@click.option('-e', '--epochs', default=5, help='Number of epochs to train the model')
@click.option('-d', '--data', default='data/ecg', help='Path to the dataset')
@click.option('-t', '--test', 'test_data', default='data/test_ecg', help='Path to the test dataset')
@click.option('-b', '--batch_size', default=16, help='Batch size for training')
@click.option('-s', '--snr', default=15, help='Signal to noise ratio for the dataset')
@click.option('-o', '--output', default='models/best.pth', help='Path to save the model')
@click.option('-a', '--amplify', default=1.0, help='Amplify factor for the STFT')
@click.option('--seed', default=42, help='Random seed')
@click.option('--logfile', default='results/log.txt', help='Path to save the log file')
@click.option('--debug', is_flag=True, help='Debug mode')
def main(epochs, data, test_data, batch_size, snr, output, amplify, seed, logfile, debug):
    
    model = ComplexUNet(128 * 128, sameW=False)
    
    stft = STFT(amplify_factor=amplify)
    
    train_set = SignalDataset(data, snr_db=snr, stft=stft)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = SignalDataset(test_data, snr_db=snr, stft=stft)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    loaders = {'train': train_loader, 'test': test_loader, 'stft': stft}
    
    trainer = Trainer(output, logfile, debug)
    trainer.train(model, loaders, epochs)
    
    
    
if __name__ == '__main__':
    main()