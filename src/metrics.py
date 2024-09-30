import torch


class PDR(torch.nn.Module):
    def __init__(self, stft):
        super().__init__()
        self.stft = stft
        

    def forward(self, y_true, y_pred):
        y_true_signal = self.stft.istft_batched(y_true)
        y_pred_signal = self.stft.istft_batched(y_pred)
        
        y_true_norm = (y_true_signal - y_true_signal.mean()) / y_true_signal.std()
        y_pred_norm = (y_pred_signal - y_pred_signal.mean()) / y_pred_signal.std()
        
        sum_ = torch.sum((y_true_norm - y_pred_norm) ** 2, dim=-1)
        prd = 100 * torch.sqrt(sum_ / torch.sum(y_true_norm ** 2, dim=-1))
        prd = prd.mean().item()
        
        # mse = torch.nn.functional.mse_loss(y_true_norm, y_pred_norm, reduction='mean')
        
        return {'prd': prd}
    
    
if __name__ == '__main__':
    
    from fourier import STFT
    
    stft = STFT()
    
    y_true = torch.rand(32, 4, 128, 128) + 1j * torch.rand(32, 4, 128, 128)
    y_pred = torch.rand(32, 4, 128, 128) + 1j * torch.rand(32, 4, 128, 128)
    
    pdr = PDR(stft)
    
    print(pdr(y_true, y_pred))