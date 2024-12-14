import click
import torch
from tqdm import tqdm

from logger import Logger
from loss import ComposedLoss, SignalMSE, SignalMAE, ComplexMSE
from fourier import STFT
from metrics import MeticEvaluator
from network import create_model
from dataset import SignalDataset


class Trainer:
    def __init__(self, best_model_fname, logfile, debug=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_model_fname = best_model_fname
        self.logger = Logger(logfile, best_model_fname)
        self.debug = debug


    def one_epoch(self, model, loader, train=True):
        model.train() if train else model.eval()
        total_loss = 0
        total_metrics = {name: 0.0 for name in self.metrics.metrics_names}
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
                model.apply(model.W_clipper)
            
            metrics_dict = self.metrics(y, y_pred)
            for name, value in metrics_dict.items():
                total_metrics[name] += value
                
        loss = total_loss / len(loader)
        for name in total_metrics:
            total_metrics[name] /= len(loader)
        return loss, total_metrics


    def train(self, model, loaders, epochs, **kwargs):
        self.logger.max_epochs = epochs
        lr = kwargs.get('lr', 1e-3)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = SignalMSE(loaders['stft'])
        self.metrics = MeticEvaluator(loaders['stft'])
        model = model.to(self.device)
        self.logger.log_model(model)
        
        for epoch in tqdm(range(epochs), leave=False, bar_format='Epoch: {l_bar}{bar:10}{r_bar}{bar:-10b}'):
            train_loss, metrics_train = self.one_epoch(model, loaders['train'])
            test_loss, metrics_test = self.one_epoch(model, loaders['test'], train=False)
            loss = {'train': train_loss, 'test': test_loss}
            metrics = {'train': metrics_train, 'test': metrics_test}
            self.logger.log(loss, metrics, epoch, model, self.best_model_fname)
            
            # if epoch == 15:
                # self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
        self.logger.draw_history()


@click.command()
@click.option('-e', '--epochs', default=5, help='Number of epochs to train the model')
@click.option('-d', '--data', default='data/ecg', help='Path to the dataset')
@click.option('-t', '--test', 'test_data', default='data/test_ecg', help='Path to the test dataset')
@click.option('-b', '--batch_size', default=8, help='Batch size for training')
@click.option('-w', '--workers', default=2, help='Number of workers for the dataloader')
@click.option('-s', '--snr', default=15, help='Signal to noise ratio for the dataset')
@click.option('-o', '--output', default='models/best.pth', help='Path to save the model')
@click.option('--seed', default=42, help='Random seed')
@click.option('--logfile', default='logs/log.txt', help='Path to save the log file')
@click.option('--debug', is_flag=True, help='Debug mode')
def main(epochs, data, test_data, batch_size, workers, snr, output, seed, logfile, debug):
    
    # model = ComplexUNet(128 * 128, sameW=False, activation='ro', diag=True)
    # model.load_weights('./models/best_new_diffW_ro.pth')
    # model.load_weights('./models/best_ro.pth')
    # model.freeze_all_except_firs_last()
    model = create_model('models/currently_best_372.pth')
    
    # model_settings = {
    #     'dimension':128*128, 
    #     'sameW':False, 
    #     'activation':'ro', 
    #     'diag':True
    # }
    # model = create_model(**model_settings)

    stft = STFT()
    
    train_set = SignalDataset(data, snr_db=[5, 20], stft=stft)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_set = SignalDataset(test_data, snr_db=[5, 20], stft=stft)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    loaders = {'train': train_loader, 'test': test_loader, 'stft': stft}
    
    trainer = Trainer(output, logfile, debug)
    trainer.train(model, loaders, epochs)
    

if __name__ == '__main__':
    main()
