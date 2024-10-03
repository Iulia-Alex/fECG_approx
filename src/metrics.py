import torch


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
        return {'prd': prd}
    
    
if __name__ == '__main__':
    
    from fourier import STFT
    
    stft = STFT()
    
    y_true = torch.rand(32, 4, 128, 128) + 1j * torch.rand(32, 4, 128, 128)
    y_pred = torch.rand(32, 4, 128, 128) + 1j * torch.rand(32, 4, 128, 128)
    
    pdr = PDR(stft)
    
    print(pdr(y_true, y_pred))