import functools
import math

import torch
import torch.nn as nn
from .block import Encoder, Decoder


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, c=None, c_out=None, n=1):
        super().__init__()
        if c is None:
            c = [3, 64, 128, 256, 512]
        if c_out is None:
            c_out = [128, 128, 128, 256, 512]
        assert len(c)==len(c_out), 'the length of c must be equal to the c_out'
        self.encoder = nn.Sequential(*[
            Encoder(c[i], c[i+1], n)
        for i in range(len(c) - 1)])
        
        self.decoder = nn.Sequential(*[
            Decoder(c_out[i], c_out[i-1], n)
        for i in range(len(c_out)-1, 0, -1)])
        self.generator=nn.Conv2d(c_out[0], c_out[0], kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.generator(x)
        return x




class OldGenerator(nn.Module):
    def __init__(self, c=4, c_out=128, inplace=False):
        super(OldGenerator, self).__init__()
        # 将所有原有的生成器层定义放到这里

        self.conv_layer1 = nn.Conv2d(c, 256, kernel_size=4, stride=2, padding=1, bias=False)

        conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        bn2_2 = nn.BatchNorm2d(256)
        # attention_2 = ResBlock_CBAM(256, 256)
        dropout2 = nn.Dropout(0.1)  # 添加Dropout层，设置丢弃比例
        conv_layer2 = [conv2, bn2_2, dropout2]
        self.conv_layer2 = nn.Sequential(*conv_layer2)
        self.relu2 = nn.ReLU(inplace=inplace)

        conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=4, padding=0, bias=False)
        bn2_3 = nn.BatchNorm2d(512)
        # attention_3 = ResBlock_CBAM(512, 512)
        relu3 = nn.ReLU(inplace=inplace)
        conv_layer3 = [conv3, bn2_3, relu3]
        self.conv_layer3 = nn.Sequential(*conv_layer3)

        conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        bn2_4 = nn.BatchNorm2d(512)
        # attention_4 = ResBlock_CBAM(512, 512)
        dropout4 = nn.Dropout(0.1)  # 添加Dropout层，设置丢弃比例
        conv_layer4 = [conv4, bn2_4, dropout4]
        self.conv_layer4 = nn.Sequential(*conv_layer4)
        self.relu4 = nn.ReLU(inplace=inplace)

        conv5 = nn.Conv2d(512, 1024, kernel_size=4, stride=4, padding=0, bias=False)
        bn2_5 = nn.BatchNorm2d(1024)
        # attention_5 = ResBlock_CBAM(1024, 1024)
        relu5 = nn.ReLU(inplace=inplace)
        conv_layer5 = [conv5, bn2_5, relu5]
        self.conv_layer5 = nn.Sequential(*conv_layer5)

        conv6 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        bn2_6 = nn.BatchNorm2d(1024)
        # attention_6 = ResBlock_CBAM(1024, 1024)
        dropout6 = nn.Dropout(0.1)  # 添加Dropout层，设置丢弃比例
        conv_layer6 = [conv6, bn2_6, dropout6]
        self.conv_layer6 = nn.Sequential(*conv_layer6)
        self.relu6 = nn.ReLU(inplace=inplace)

        conv11 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        # attention_11 = ResBlock_CBAM(1024, 1024)
        relu11 = nn.ReLU(inplace=inplace)
        trans1 = [conv11, relu11]
        self.trans1 = nn.Sequential(*trans1)
        deconv11 = nn.ConvTranspose3d(512, 512, kernel_size=1, stride=1, padding=0, output_padding=0, bias=False)
        relu12 = nn.ReLU(inplace=inplace)
        trans2 = [deconv11, relu12]
        self.trans2 = nn.Sequential(*trans2)

        deconv7 = nn.ConvTranspose3d(512, 256, kernel_size=4, stride=4, padding=0, output_padding=0, bias=False)
        bn3_7 = nn.BatchNorm3d(256)
        relu16 = nn.ReLU(inplace=inplace)
        deconv_layer7 = [deconv7, bn3_7, relu16]
        self.deconv_layer7 = nn.Sequential(*deconv_layer7)

        deconv6 = nn.ConvTranspose3d(256, 256, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False)
        bn3_6 = nn.BatchNorm3d(256)
        dropout16 = nn.Dropout(0.1)  # 添加Dropout层，设置丢弃比例
        relu17 = nn.ReLU(inplace=inplace)
        deconv_layer6 = [deconv6, bn3_6, dropout16, relu17]
        self.deconv_layer6 = nn.Sequential(*deconv_layer6)

        deconv5 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=4, padding=0, output_padding=0, bias=False)
        bn3_5 = nn.BatchNorm3d(128)
        dropout17 = nn.Dropout(0.1)  # 添加Dropout层，设置丢弃比例
        relu17 = nn.ReLU(inplace=inplace)
        deconv_layer5 = [deconv5, bn3_5, dropout17, relu17]
        self.deconv_layer5 = nn.Sequential(*deconv_layer5)

        deconv4 = nn.ConvTranspose3d(128, 128, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False)
        bn3_4 = nn.BatchNorm3d(128)
        dropout18 = nn.Dropout(0.1)  # 添加Dropout层，设置丢弃比例
        relu18 = nn.ReLU(inplace=inplace)
        deconv_layer4 = [deconv4, bn3_4, dropout18, relu18]
        self.deconv_layer4 = nn.Sequential(*deconv_layer4)

        deconv3 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        bn3_3 = nn.BatchNorm3d(64)
        dropout19 = nn.Dropout(0.1)  # 添加Dropout层，设置丢弃比例
        relu19 = nn.ReLU(inplace=inplace)
        deconv_layer3 = [deconv3, bn3_3, dropout19, relu19]
        self.deconv_layer3 = nn.Sequential(*deconv_layer3)

        deconv2 = nn.ConvTranspose3d(64, 64, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False)
        bn3_2 = nn.BatchNorm3d(64)
        dropout20 = nn.Dropout(0.1)  # 添加Dropout层，设置丢弃比例
        relu20 = nn.ReLU(inplace=inplace)
        deconv_layer2 = [deconv2, bn3_2, dropout20, relu20]
        self.deconv_layer2 = nn.Sequential(*deconv_layer2)

        deconv1 = nn.Conv3d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)
        relu21 = nn.ReLU(inplace=inplace)
        deconv_layer1 = [deconv1, relu21]
        self.deconv_layer1 = nn.Sequential(*deconv_layer1)

        self.output_layer = nn.Conv2d(64, c_out, kernel_size=1, stride=1, padding=0, bias=False)

        initialize_weights(self)

    def forward(self, x):
        # 将原有的生成器 forward 方法中的代码放到这里
        conv1 = self.conv_layer1(x)
        conv2 = self.conv_layer2(conv1)
        relu2 = self.relu2(conv1 + conv2)

        conv3 = self.conv_layer3(relu2)
        conv4 = self.conv_layer4(conv3)
        relu4 = self.relu4(conv3 + conv4)

        conv5 = self.conv_layer5(relu4)
        conv6 = self.conv_layer6(conv5)
        relu6 = self.relu6(conv5 + conv6)

        features = self.trans1(relu6)
        trans_features = features.view(-1, 512, 2, 4, 4)
        trans_features = self.trans2(trans_features)

        deconv7 = self.deconv_layer7(trans_features)
        deconv6 = self.deconv_layer6(deconv7)
        deconv5 = self.deconv_layer5(deconv6)
        deconv4 = self.deconv_layer4(deconv5)
        deconv3 = self.deconv_layer3(deconv4)
        deconv2 = self.deconv_layer2(deconv3)
        deconv1 = self.deconv_layer1(deconv2)

        out = torch.squeeze(deconv1, 1)
        out = self.output_layer(out)

        return out


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Dropout(0.1)  # Adding Dropout layer with dropout probability of 0.35
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1)  # Adding Dropout layer with dropout probability of 0.35
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PatchGAN(nn.Module):
    def __init__(self, c=4, c_out=128, n=None, is_new=False):
        super(PatchGAN, self).__init__()
        self.generator = Generator(c, c_out, n) if is_new else OldGenerator(c, c_out)
        self.discriminator = NLayerDiscriminator(input_nc=c_out[0] if is_new else c_out) 
        initialize_weights(self)

    def forward(self, x):
        gen_out = self.generator(x)
        # 将生成器的输出传递给判别器
        disc_out = self.discriminator(gen_out)
        return gen_out, disc_out