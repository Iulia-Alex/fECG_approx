import torch



class CustomLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    
    def check_input(self, input_):
        if torch.isnan(input_).any():
            raise ValueError('Input contains NaN')
    
    def compute_loss_val(self, loss):
        if torch.isnan(loss).all():
            raise ValueError('Loss is NaN')
        
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        loss = torch.mean(loss)
        return loss


class ComplexMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y, y_pred):
        y_real, y_imag = torch.real(y), torch.imag(y)
        y_pred_real, y_pred_imag = torch.real(y_pred), torch.imag(y_pred)
        return torch.mean((y_real - y_pred_real) ** 2 + (y_imag - y_pred_imag) ** 2)


class ComplexMAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y, y_pred):
        y_real, y_imag = torch.real(y), torch.imag(y)
        y_pred_real, y_pred_imag = torch.real(y_pred), torch.imag(y_pred)
        return torch.mean(torch.abs(y_real - y_pred_real) + torch.abs(y_imag - y_pred_imag))
    
    
class SignalMSE(CustomLoss):
    def __init__(self, stft):
        super().__init__()
        self.stft = stft
        
    def forward(self, y_true, y_pred, signal=False):
        self.check_input(y_pred)
        if not signal:
            y_true = self.stft.istft_batched(y_true)
            y_pred = self.stft.istft_batched(y_pred)
        mse = torch.nn.functional.mse_loss(y_true, y_pred, reduction='mean')
        mse = self.compute_loss_val(mse)
        return mse
    
    
class SignalMAE(CustomLoss):
    def __init__(self, stft):
        super().__init__()
        self.stft = stft
        
    def forward(self, y_true, y_pred, signal=False):
        self.check_input(y_pred)
        if not signal:
            y_true = self.stft.istft_batched(y_true)
            y_pred = self.stft.istft_batched(y_pred)
        mae = torch.nn.functional.l1_loss(y_true, y_pred, reduction='none')
        mae = self.compute_loss_val(mae)
        return mae
    
    
class ComposedLoss(torch.nn.Module):
    def __init__(self, stft, kind='mse'):
        super().__init__()
        self.stft = stft
        
        if kind in ['mae', 'l1']:
            self.loss_signal = SignalMAE(stft)
            self.loss_spec = ComplexMAE()
        
        elif kind in ['mse', 'l2']:
            self.loss_signal = SignalMSE(stft)
            self.loss_spec = ComplexMSE()
            
        else:
            raise ValueError(f'Unknown loss kind: {kind}')


    def forward(self, y_true, y_pred):
        loss_signal = self.loss_signal(y_true, y_pred)
        loss_spec = self.loss_spec(y_true, y_pred)
        return loss_signal + loss_spec 
        