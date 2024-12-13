import torch

from scipy.stats import pearsonr


class PDR(torch.nn.Module):
    def __init__(self, stft):
        super().__init__()
        self.stft = stft
        

    def forward(self, y_true, y_pred, signal=False):
        if not signal:
            y_true = self.stft.istft_batched(y_true)
            y_pred = self.stft.istft_batched(y_pred)
        
        y_true_norm = (y_true - y_true.mean()) / y_true.std()
        y_pred_norm = (y_pred - y_pred.mean()) / y_pred.std()
        
        sum_ = torch.sum((y_true_norm - y_pred_norm) ** 2, dim=-1)
        prd = 100 * torch.sqrt(sum_ / torch.sum(y_true_norm ** 2, dim=-1))
        prd = prd.mean().item()        
        return prd
    

class PCC(torch.nn.Module):
    def __init__(self, stft):
        super().__init__()
        self.stft = stft
    
    def forward(self, y_true, y_pred, signal=False):
        if not signal:
            y_true = self.stft.istft_batched(y_true)
            y_pred = self.stft.istft_batched(y_pred)
        
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        
        pcc = pearsonr(y_true.flatten(), y_pred.flatten())[0]
        return float(pcc)




class MeticEvaluator:
    def __init__(self, stft, metric_list=['pdr', 'pcc']):
        self.stft = stft
        self.metrics = self._get_metrics(metric_list)
        
    
    def __call__(self, y_true, y_pred, signal=False):
        results = {}
        for metric_name, metric in self.metrics.items():
            results[metric_name] = metric(y_true, y_pred, signal)
        return results
        
    def _get_metrics(self, metric_list):
        metrics = {}
        for metric in metric_list:
            if metric == 'pdr':
                metrics[metric] = PDR(self.stft)
            elif metric == 'pcc':
                metrics[metric] = PCC(self.stft)
            else:
                raise ValueError(f'Unknown metric: {metric}')
        return metrics
        


if __name__ == '__main__':
    
    from fourier import STFT
    
    stft = STFT()
    
    y_true = torch.rand(32, 4, 128, 128) + 1j * torch.rand(32, 4, 128, 128)
    y_pred = torch.rand(32, 4, 128, 128) + 1j * torch.rand(32, 4, 128, 128)
    
    pdr = PDR(stft)
    
    print(pdr(y_true, y_pred))
    
    pcc = PCC(stft)
    
    print(pcc(y_true, y_pred), pcc(y_true, y_true + 1e-3))