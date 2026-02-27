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


class GKActivation(ComplexLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / (1 + torch.abs(x))


class SquashActivation(ComplexLayer):
    def __init__(self):
        super().__init__()
        self.c = 8.0 / (3.0 * torch.sqrt(torch.tensor(3.0)))

    def forward(self, x):
        return self.c * (x / (1 + torch.abs(x) ** 2))
    

class SmeqActivation(ComplexLayer):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        real_x, imag_x = self.extract_real_imag(x)
        min_ = torch.min(real_x, imag_x)
        max_ = torch.max(real_x, imag_x)
        return self.combine(min_, max_)
        

class RoActivation(ComplexLayer):
    def __init__(self):
        super().__init__()
        self.mu_logits = torch.nn.Parameter(torch.randn(3))
        self.CReLU = ComplexReLU()
        self.GK = GKActivation()
        self.Smeq = SmeqActivation()
        
    def forward(self, x):
        mu = torch.nn.functional.softmax(self.mu_logits, dim=0)
        x = mu[0] * self.CReLU(x) + mu[1] * self.GK(x) + mu[2] * self.Smeq(x)
        return x


class ComplexConvLayer(ComplexLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        activation=SquashActivation,
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
            self.norm = nn.BatchNorm2d(out_channels)
            
        else:
            self.conv_real = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding
            )
            self.conv_imag = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding
            )
            self.norm_real = nn.BatchNorm2d(out_channels)
            self.norm_imag = nn.BatchNorm2d(out_channels)

    # overide in order to preserve proprieties of convolution on complex numbers
    def combine(self, real, imag):
        return (real - imag) + 1j * (real + imag)
    

    def forward(self, x):
        x_real, x_imag = self.extract_real_imag(x)
        if self.sameW:
            x_real = self.conv(x_real)
            x_real = self.norm(x_real)
            x_imag = self.conv(x_imag)
            x_imag = self.norm(x_imag)
        else:
            x_real = self.conv_real(x_real)
            x_real = self.norm_real(x_real)
            x_imag = self.conv_imag(x_imag)
            x_imag = self.norm_imag(x_imag)
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
        x_real = x_real.view(b * c, h * w)
        x_imag = x_imag.view(b * c, h * w)

        if self.sameW:
            x_real = x_real @ torch.diag(torch.exp(self.betas))
            x_imag = x_imag @ torch.diag(torch.exp(self.betas))
        else:
            x_real = x_real @ torch.diag(torch.exp(self.betas_real))
            x_imag = x_imag @ torch.diag(torch.exp(self.betas_imag))

        x_real = x_real.view(b, c, h, w)
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
        activation=SquashActivation,
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
        activation=SquashActivation,
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


class WeightClipper:
    def __init__(self, clip_value=1):
        self.clip_value = clip_value

    def __call__(self, module):
        if hasattr(module, "weight"):
            w = module.weight.data
            w = torch.clamp(w, -self.clip_value, self.clip_value)
            module.weight.data = w


class ComplexUNet(nn.Module):
    def __init__(self, dimension, sameW=False, activation='crelu', diag=False, clip_value=1):
        super().__init__()
        
        match activation:
            case 'crelu': self.act = ComplexReLU
            case 'gk': self.act = GKActivation
            case 'squash': self.act = SquashActivation
            case 'ro': self.act = RoActivation
            case _: raise ValueError(f"Unknown activation: {activation}")
    
        self.metadata = {
            'dimension': dimension,
            'sameW': sameW,
            'activation': activation,
            'diag': diag,
        }
        
        self.W_clipper = WeightClipper(clip_value)
        
        self.diag = diag
        if diag:
            self.diag_in = Diag(dimension, sameW=sameW)
            self.diag_out = Diag(dimension, sameW=sameW)
        
        self.conv1 = ComplexConvLayer(1, 64, sameW=sameW, activation=self.act)
        
        
        self.down1 = ComplexDownBlock(64, 128, sameW=sameW, activation=self.act)
        self.down2 = ComplexDownBlock(128, 256, sameW=sameW, activation=self.act)
        self.down3 = ComplexDownBlock(256, 512, sameW=sameW, activation=self.act)
        
        self.bottleneck = nn.Sequential(
            ComplexConvLayer(512, 1024, sameW=sameW, activation=self.act),
            ComplexConvLayer(1024, 1024, sameW=sameW, activation=self.act),
            ComplexConvLayer(1024, 512, sameW=sameW, activation=self.act),
        )

        self.up1 = ComplexUpBlock(1024, 256, sameW=sameW, activation=self.act)
        self.up2 = ComplexUpBlock(512, 128, sameW=sameW, activation=self.act)
        self.up3 = ComplexUpBlock(256, 64, sameW=sameW, activation=self.act)
        
        self.conv2 = ComplexConvLayer(64, 32, sameW=sameW, activation=self.act)
        self.conv3 = ComplexConvLayer(32, 1, kernel_size=1, padding=0, sameW=sameW, activation=self.act)
        self.sigma = nn.Sigmoid()

        self.out = torch.nn.Sequential(
            # ComplexConvLayer(2, 1, sameW=sameW, activation=self.act),
            ComplexConvLayer(1, 1, sameW=sameW, activation=self.act),
            ComplexConvLayer(1, 1, kernel_size=1, padding=0, sameW=sameW, activation=self.act),
            # ComplexConvLayer(1, 1, kernel_size=1, padding=0, sameW=sameW, activation=self.act),
        )


    def normalize(self, x):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        self.mean = mean
        self.std = std
        return (x - mean) / (std + 1e-6)


    def denormalize(self, x):
        return x * self.std + self.mean


    def forward(self, x):
        b, c, h, w = x.size()
        # x = x.view(b * c, 1, h, w)  # now we have 1 channel only
        
        x = self.normalize(x)
        init = x

        if self.diag: 
            x = self.diag_in(x)

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

        if self.diag: 
            x = self.diag_out(x)

        # x = self.sigma(x) * init
        # magnitude = torch.abs(x)
        # mask = torch.sigmoid(magnitude)  # mask e real, intre 0 si 1
        # x = mask * init  # aplica masca pe spectrograma originala

        # in forward() din network.py
        x_cpu = x.cpu()
        x_real, x_imag = torch.real(x_cpu), torch.imag(x_cpu)
        mask = torch.sigmoid(x_real) + 1j * torch.sigmoid(x_imag)
        print(f"Mask: real min={torch.real(mask).min():.4f}, max={torch.real(mask).max():.4f}, mean={torch.real(mask).mean():.4f}")
        x = mask.to(x.device) * init

        x = self.denormalize(x)
        # x = x.view(b, c, h, w)  # now we work with only one channel
        return x

    # custom weights initialization
    def load_weights(self, path):
        w = torch.load(path, map_location='cpu')
        for name, param in self.named_parameters():
            if name in w:
                if param.data.size() == w[name].size():
                    param.data = w[name]
        del w


    def freeze_all_except_firs_last(self):
        for name, param in self.named_parameters():
            if "conv1" in name or "out" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        

    def unfreeze_all(self):
        for name, param in self.named_parameters():
            param.requires_grad = True





def create_model(ckpt_path=None, **kwargs):
    if ckpt_path is None:
        print(f'[INFO] Creating model with parameters: {kwargs} since no checkpoint was provided')
        return ComplexUNet(**kwargs)
    
    pack = torch.load(ckpt_path, map_location='cpu')
    if 'metadata' in pack.keys():  # new packs with metadata included
        metadata = pack['metadata']
        model = ComplexUNet(**metadata)
        pack = pack['state_dict']
        print(f'[INFO] Model loaded with metadata: {metadata} from {ckpt_path}')
    else:  # old packs, recreate model with given parameters
        model = ComplexUNet(**kwargs)
        print(f'[INFO] Model created since no metadata was found in {ckpt_path}')
    model.load_state_dict(pack)
    
    return model
    


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    b, c, h, w = 8, 1, 128, 128
    x = torch.randn(b, c, h, w) + 1j * torch.randn(b, c, h, w)
    x = x.to(device)
    
    #### Old model without metadatas
    # model = create_model('models/best_new_diffW_ro_v2.pth', dimension=h * w, sameW=False, activation='ro', diag=True)
    # model = ComplexUNet(h * w, sameW=True, activation='ro', diag=True)
    # model = model.to(device)

    # params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters: {params / 1e6:.2f} M")

    # y = model(x)
    # print("output:", y.size())

    # metadata = model.metadata
    # print(metadata)
    
    # pack = {
        # 'metadata': metadata,
        # 'state_dict': model.state_dict()
    # }
    
    # torch.save(pack, 'models/with_metadata/best_new_diffW_ro_v2.pth')
    
    
    ##### new model with metadata
    model = create_model('models/with_metadata/best_new_diffW_ro_v2.pth')
    print(f'Model loaded with metadata: {model.metadata}')