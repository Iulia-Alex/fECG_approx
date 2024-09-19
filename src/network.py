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
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        activation=ComplexReLU,
        sameW=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation()
        self.sameW = sameW

        if self.sameW:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding
            )
        else:
            self.conv_real = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding
            )
            self.conv_imag = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding
            )

    def forward(self, x):
        x_real, x_imag = self.extract_real_imag(x)
        if self.sameW:
            x_real = self.conv(x_real)
            x_imag = self.conv(x_imag)
        else:
            x_real = self.conv_real(x_real)
            x_imag = self.conv_imag(x_imag)
        u = self.activation(self.combine(x_real, x_imag))
        return u


class Diag(ComplexLayer):
    def __init__(self, dimension, sameW=False):
        super().__init__()
        self.sameW = sameW

        if self.sameW:
            self.betas = nn.Parameter(torch.ones(dimension))
        else:
            self.betas_real = nn.Parameter(torch.ones(dimension))
            self.betas_imag = nn.Parameter(torch.ones(dimension))

    def forward(self, x):
        x_real, x_imag = self.extract_real_imag(x)

        b, c, h, w = x_real.size()
        x_real = x_real.contiguous().view(b * c, h * w)
        x_imag = x_imag.contiguous().view(b * c, h * w)

        if self.sameW:
            x_real = x_real @ torch.diag(torch.exp(self.betas))
            x_imag = x_imag @ torch.diag(torch.exp(self.betas))
        else:
            x_real = x_real @ torch.diag(torch.exp(self.betas_real))
            x_imag = x_imag @ torch.diag(torch.exp(self.betas_imag))

        x_real = x_real.contiguous().view(b, c, h, w)
        x_imag = x_imag.contiguous().view(b, c, h, w)

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
        self.upsampler = nn.Upsample(scale_factor=scale_factor, mode="nearest")

    def forward(self, x):
        x_real, x_imag = self.extract_real_imag(x)
        x_real = self.upsampler(x_real)
        x_imag = self.upsampler(x_imag)
        return self.combine(x_real, x_imag)


class ComplexDownBlock(ComplexLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        activation=ComplexReLU,
        scale_factor=2,
        sameW=False,
    ):
        super().__init__()
        self.conv1 = ComplexConvLayer(
            in_channels, out_channels, kernel_size, stride, padding, activation, sameW
        )
        self.conv2 = ComplexConvLayer(
            out_channels, out_channels, kernel_size, stride, padding, activation, sameW
        )
        self.conv3 = ComplexConvLayer(
            out_channels, out_channels, kernel_size, stride, padding, activation, sameW
        )
        self.down = ComplexDownSample(scale_factor)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.down(x)


class ComplexUpBlock(ComplexLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        activation=ComplexReLU,
        scale_factor=2,
        sameW=False,
    ):
        super().__init__()
        self.conv1 = ComplexConvLayer(
            in_channels, out_channels, kernel_size, stride, padding, activation, sameW
        )
        self.conv2 = ComplexConvLayer(
            out_channels, out_channels, kernel_size, stride, padding, activation, sameW
        )
        self.conv3 = ComplexConvLayer(
            out_channels, out_channels, kernel_size, stride, padding, activation, sameW
        )
        self.up = ComplexUpSample(scale_factor)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.up(x)


class ComplexUNet(nn.Module):
    def __init__(self, dimension, sameW=False):
        super().__init__()
        self.diag1 = Diag(dimension, sameW=sameW)
        self.conv1 = ComplexConvLayer(4, 64, sameW=sameW)
        
        
        self.down1 = ComplexDownBlock(64, 128, sameW=sameW)
        self.down2 = ComplexDownBlock(128, 256, sameW=sameW)
        self.down3 = ComplexDownBlock(256, 512, sameW=sameW)
        
        self.bottleneck = nn.Sequential(
            ComplexConvLayer(512, 1024, sameW=sameW),
            ComplexConvLayer(1024, 1024, sameW=sameW),
            ComplexConvLayer(1024, 512, sameW=sameW),
        )

        self.up1 = ComplexUpBlock(1024, 256, sameW=sameW)
        self.up2 = ComplexUpBlock(512, 128, sameW=sameW)
        self.up3 = ComplexUpBlock(256, 64, sameW=sameW)
        
        self.conv2 = ComplexConvLayer(64, 64, sameW=sameW)
        self.conv3 = ComplexConvLayer(64, 4, kernel_size=1, padding=0, sameW=sameW)
        self.out = Diag(dimension, sameW=sameW)
        self.sigma = nn.Tanh()


    def forward(self, x):
        init = x
        x = self.diag1(x)
        x = self.conv1(x)
        res1 = self.down1(x)
        res2 = self.down2(res1)
        res3 = self.down3(res2)
        
        x = self.bottleneck(res3)

        x = self.up1(torch.cat([x, res3], dim=1))
        x = self.up2(torch.cat([x, res2], dim=1))
        x = self.up3(torch.cat([x, res1], dim=1))
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.out(x)
        x = self.sigma(x)
        x = init * x
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    b, c, h, w = 16, 4, 128, 128
    x = torch.randn(b, c, h, w) + 1j * torch.randn(b, c, h, w)
    x = x.to(device)

    model = ComplexUNet(h * w, sameW=True)
    model = model.to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {params / 1e6:.2f} M")

    y = model(x)
    print("output:", y.size())
