import torch
import torch.nn as nn


class ComplexLayer(nn.Module):
    """Wrapper for complex layers, with utility functions."""
    def __init__(self):
        super().__init__()
        
    def extract_real_imag(self, x):
        return torch.real(x), torch.imag(x)
    
    def combine(self, real, imag):
        return real + 1j * imag



class ComplexReLU(ComplexLayer):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        x_real, x_imag = self.extract_real_imag(x)
        x_real = self.relu(x_real)
        x_imag = self.relu(x_imag)
        return self.combine(x_real, x_imag)

class ComplexConvLayer(ComplexLayer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=ComplexReLU):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x_real, x_imag = self.extract_real_imag(x)
        x_real = self.conv(x_real)
        x_imag = self.conv(x_imag)
        u = self.activation(self.combine(x_real, x_imag))
        return u


class Diag(ComplexLayer):
    def __init__(self, dimension):
        super().__init__()
        self.betas = nn.Parameter(torch.ones(dimension))

    def forward(self, x):
        x_real, x_imag = self.extract_real_imag(x)
        
        b, c, h, w = x_real.size()
        x_real = x_real.view(b * c, h * w)
        x_real = x_real @ torch.diag(torch.exp(self.betas))
        x_real = x_real.view(b, c, h, w)
        
        x_imag = x_imag.view(b * c, h * w)
        x_imag = x_imag @ torch.diag(torch.exp(self.betas))
        x_imag = x_imag.view(b, c, h, w)
        
        return self.combine(x_real, x_imag)


class ComplexDownSample(ComplexLayer):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.downsampler = nn.MaxPool2d(scale_factor)
        
    def forward(self, x):
        x_real, x_imag = self.extract_real_imag(x)
        x_real = self.downsampler(x_real)
        x_imag = self.downsampler(x_imag)
        return self.combine(x_real, x_imag)
    
    
class ComplexUpSample(ComplexLayer):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsampler = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        
    def forward(self, x):
        x_real, x_imag = self.extract_real_imag(x)
        x_real = self.upsampler(x_real)
        x_imag = self.upsampler(x_imag)
        return self.combine(x_real, x_imag)


class ComplexDownBlock(ComplexLayer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=ComplexReLU, scale_factor=2):
        super().__init__()
        self.conv = ComplexConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation)
        self.down = ComplexDownSample(scale_factor)
    
    def forward(self, x):
        return self.down(self.conv(x))


class ComplexUpBlock(ComplexLayer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=ComplexReLU, scale_factor=2):
        super().__init__()
        self.conv = ComplexConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation)
        self.up = ComplexUpSample(scale_factor)
    
    def forward(self, x):
        return self.up(self.conv(x))
    


class ComplexUNet(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.diag1 = Diag(dimension)
        self.down1 = ComplexDownBlock(4, 64)
        self.down2 = ComplexDownBlock(64, 128)
        self.down3 = ComplexDownBlock(128, 256)
        
        self.bottleneck = ComplexDownBlock(256, 256)
        
        self.up1 = ComplexUpBlock(256, 256)
        self.up2 = ComplexUpBlock(256, 128)
        self.up3 = ComplexUpBlock(128, 64)
        self.up4 = ComplexUpBlock(64, 4)
        
        
        self.out = Diag(dimension)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.diag1(x)
        res1 = self.down1(x)
        res2 = self.down2(res1)
        res3 = self.down3(res2)
        
        x = self.bottleneck(res3)
        
        x = self.up1(x) + res3
        x = self.up2(x) + res2
        x = self.up3(x) + res1
        x = self.up4(x)
        
        x = self.out(x)
        # x = self.sigmoid(x)
        return x
        
    
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    b, c, h, w = 32, 4, 128, 128
    x = torch.randn(b, c, h, w) + 1j * torch.randn(b, c, h, w)
    x = x.to(device)

    
    model = ComplexUNet(h * w)
    model = model.to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {params / 1e6:.2f} M')
    
    y = model(x)
    print('output:', y.size())
