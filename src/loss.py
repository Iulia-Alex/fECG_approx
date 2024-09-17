import torch


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