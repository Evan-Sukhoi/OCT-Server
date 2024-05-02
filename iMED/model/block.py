import torch
from torch import nn
from .conv import Conv, ConvTrans


class Bottleneck(nn.Module):

    def __init__(self, c1, c2, act=True, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1, act=act)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class ChannelAttentionModule(nn.Module):
    def __init__(self, c, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(c, c // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(c // ratio, c, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, c):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(c)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class ResBlock_CBAM(nn.Module):
    def __init__(self, c1, c2, n=1, downsampling=False):
        super(ResBlock_CBAM, self).__init__()
        self.downsampling = downsampling

        self.bottleneck = nn.ModuleList(Bottleneck(c1, c1, k=((3, 3), (3, 3)), e=1.0, act=nn.ReLU()) for _ in range(n))
        self.conv = Conv(n * c1, c2)
        self.cbam = CBAM(c=c2)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(c2)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = list(m(x) for m in self.bottleneck)
        out = self.conv(torch.cat(out, 1))
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Encoder(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c2 = c2
        self.feature_extract = C2f(c1, c2 // 2, n=n, e=e, shortcut=shortcut, g=g)
        self.attention = ResBlock_CBAM(c1, c2 // 2, n=n, downsampling=True if c1 != c2 // 2 else False)
        self.downsampling = Conv((c2 // 2) * 2, c2, k=3, s=2, p=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        b, _, h, w = x.shape
        out = torch.zeros((b, (self.c2 // 2) * 2, h, w)).to(x.device)
        out[:, 0:self.c2 // 2, :, :] = self.feature_extract(x)
        out[:, self.c2 // 2:(self.c2 // 2) * 2, :, :] = self.attention(x)
        out = self.downsampling(out)
        out = self.act(out)
        return out


class Decoder(nn.Module):

    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.upsampling = ConvTrans(c1, c2)
        self.attention = ResBlock_CBAM(c2, c2, n=n)

    def forward(self, x):
        x = self.upsampling(x)
        x = self.attention(x)
        return x
