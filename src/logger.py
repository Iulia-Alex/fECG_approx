import os
import torch
from tqdm import tqdm
from datetime import datetime as dt


class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        log_folder = os.path.dirname(log_path)
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        with open(log_path, 'w') as f:
            f.write('')
        
        self.get_time = lambda: dt.now()
        self.current_time = self.get_time()
        
        self.best_loss = float('inf')
        self.history = {'train': [], 'test': []}


    def draw_history(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.history['train'], label='train')
        plt.plot(self.history['test'], 'r', label='test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'results/history_{self.best_model_fname}.png')
        plt.close()
    

    def log_info(self, msg):
        with open(self.log_path, 'a') as f:
            f.write(msg + '\n')
        tqdm.write(msg)
        
        
    def log(self, train_loss, test_loss, epoch, model, save_path):
        text = f'[Epoch {epoch + 1}/{self.max_epochs} ]\n'
        text += f'\tTrain loss: {train_loss:.3e}\n'
        text += f'\tTest loss: {test_loss:.3e}\n'
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            torch.save(model.state_dict(), save_path)
            text += '\tModel saved\n'
        elapsed_time = self.get_time() - self.current_time
        self.current_time = self.get_time()
        text += f'Elapsed time: {elapsed_time.total_seconds():.2f} s'
        text += '\n\n'
        self.history['train'].append(train_loss)
        self.history['test'].append(test_loss)
        self.log_info(text)