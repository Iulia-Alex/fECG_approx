import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime as dt


class Logger:
    def __init__(self, log_path=None):
        self.log_path = log_path
        if log_path is not None:
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
        
    
    def draw_from_file(self, path):
        raise NotImplementedError('This method is not implemented yet')
        

    def __check_log_path(self):
        if self.log_path is None:
            raise ValueError('[ERROR] Log path is not defined, please set the log_path attribute')

    def log_info(self, msg):
        self.__check_log_path()
        with open(self.log_path, 'a') as f:
            f.write(msg + '\n')
        tqdm.write(msg)
        
        
    def log(self, loss, metrics, epoch, model, save_path):
        train_loss, test_loss = loss['train'], loss['test']
        metrics_train, metrics_test = metrics['train'], metrics['test']
        
        self.__check_log_path()
        text = f'[Epoch {epoch + 1}/{self.max_epochs} ]\n'
        text += f'\tTrain: loss: {train_loss:.3e} | prd: {metrics_train["prd"]:.3f}\n'
        text += f'\tTest: loss: {test_loss:.3e} | prd: {metrics_test["prd"]:.3f}\n'
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
        


# TODO: Implement the main function, which will be used to draw from the log file
def main():
    pass


if __name__ == '__main__':
    main()