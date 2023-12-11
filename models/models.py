import math
import numpy as np

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels=3, depth=3):
        super(Encoder, self).__init__()

        # Shuffle pixels to expand in channel dimension
        # shuffler_list = [PixelShuffle(0.5) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        self.shuffler = PixelShuffle(1 / 2**depth)

        relu = nn.LeakyReLU(0.2, True)
        
        # FF_RCAN or FF_Resblocks
        self.interpolate = Interpolation(5, 12, in_channels * (4**depth), act=relu)
        
    def forward(self, x1, x2):
        """
        Encoder: Shuffle-spread --> Feature Fusion --> Return fused features
        """
        feats1 = self.shuffler(x1)
        feats2 = self.shuffler(x2)

        feats = self.interpolate(feats1, feats2)

        return feats


class Decoder(nn.Module):
    def __init__(self, depth=3):
        super(Decoder, self).__init__()

        # shuffler_list = [PixelShuffle(2) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        self.shuffler = PixelShuffle(2**depth)

    def forward(self, feats):
        out = self.shuffler(feats)
        return out


class CAIN(nn.Module):
    def __init__(self,training, depth=3):
        super(CAIN, self).__init__()
        
        self.encoder = Encoder(in_channels=3, depth=depth)
        self.decoder = Decoder(depth=depth)
        self.training = training
    def forward(self, x1, x2):
        x1, m1 = sub_mean(x1)
        x2, m2 = sub_mean(x2)

        if not self.training:
            paddingInput, paddingOutput = InOutPaddings(x1)
            x1 = paddingInput(x1)
            x2 = paddingInput(x2)

        feats = self.encoder(x1, x2)
        out = self.decoder(feats)

        if not self.training:
            out = paddingOutput(out)

        mi = (m1 + m2) / 2
        out += mi

        return out, feats


import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def sub_mean(x):
    mean = x.mean(2, keepdim=True).mean(3, keepdim=True)
    x -= mean
    return x, mean

def InOutPaddings(x):
    w, h = x.size(3), x.size(2)
    padding_width, padding_height = 0, 0
    if w != ((w >> 7) << 7):
        padding_width = (((w >> 7) + 1) << 7) - w
    if h != ((h >> 7) << 7):
        padding_height = (((h >> 7) + 1) << 7) - h
    paddingInput = nn.ReflectionPad2d(padding=[padding_width // 2, padding_width - padding_width // 2,
                                               padding_height // 2, padding_height - padding_height // 2])
    paddingOutput = nn.ReflectionPad2d(padding=[0 - padding_width // 2, padding_width // 2 - padding_width,
                                                0 - padding_height // 2, padding_height // 2 - padding_height])
    return paddingInput, paddingOutput


class ConvNorm(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride=1, norm=False):
        super(ConvNorm, self).__init__()

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_feat, out_feat, stride=stride, kernel_size=kernel_size, bias=True)

        self.norm = norm
        if norm == 'IN':
            self.norm = nn.InstanceNorm2d(out_feat, track_running_stats=True)
        elif norm == 'BN':
            self.norm = nn.BatchNorm2d(out_feat)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        return out


class UpConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, mode='transpose', norm=False):
        super(UpConvNorm, self).__init__()

        if mode == 'transpose':
            self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        elif mode == 'shuffle':
            self.upconv = nn.Sequential(
                ConvNorm(in_channels, 4*out_channels, kernel_size=3, stride=1, norm=norm),
                PixelShuffle(2))
        else:
            # out_channels is always going to be the same as in_channels
            self.upconv = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                ConvNorm(in_channels, out_channels, kernel_size=1, stride=1, norm=norm))
    
    def forward(self, x):
        out = self.upconv(x)
        return out



class meanShift(nn.Module):
    def __init__(self, rgbRange, rgbMean, sign, nChannel=3):
        super(meanShift, self).__init__()
        if nChannel == 1:
            l = rgbMean[0] * rgbRange * float(sign)

            self.shifter = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
            self.shifter.weight.data = torch.eye(1).view(1, 1, 1, 1)
            self.shifter.bias.data = torch.Tensor([l])
        elif nChannel == 3:  
            r = rgbMean[0] * rgbRange * float(sign)
            g = rgbMean[1] * rgbRange * float(sign)
            b = rgbMean[2] * rgbRange * float(sign)

            self.shifter = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
            self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
            self.shifter.bias.data = torch.Tensor([r, g, b])
        else:
            r = rgbMean[0] * rgbRange * float(sign)
            g = rgbMean[1] * rgbRange * float(sign)
            b = rgbMean[2] * rgbRange * float(sign)
            self.shifter = nn.Conv2d(6, 6, kernel_size=1, stride=1, padding=0)
            self.shifter.weight.data = torch.eye(6).view(6, 6, 1, 1)
            self.shifter.bias.data = torch.Tensor([r, g, b, r, g, b])

        # Freeze the meanShift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)

        return x


""" CONV - (BN) - RELU - CONV - (BN) """
class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size=3, reduction=False, bias=True, # 'reduction' is just for placeholder
                 norm=False, act=nn.ReLU(True), downscale=False):
        super(ResBlock, self).__init__()

        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size=kernel_size, stride=2 if downscale else 1),
            act,
            ConvNorm(out_feat, out_feat, kernel_size=kernel_size, stride=1)
        )
        
        self.downscale = None
        if downscale:
            self.downscale = nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=2)

    def forward(self, x):
        res = x
        out = self.body(x)
        if self.downscale is not None:
            res = self.downscale(res)
        out += res

        return out 


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y, y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, reduction, bias=True,
            norm=False, act=nn.ReLU(True), downscale=False, return_ca=False):
        super(RCAB, self).__init__()

        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size, stride=2 if downscale else 1, norm=norm),
            act,
            ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
            CALayer(out_feat, reduction)
        )
        self.downscale = downscale
        if downscale:
            self.downConv = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1)
        self.return_ca = return_ca

    def forward(self, x):
        res = x
        out, ca = self.body(x)
        if self.downscale:
            res = self.downConv(res)
        out += res

        if self.return_ca:
            return out, ca
        else:
            return out


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, Block, n_resblocks, n_feat, kernel_size, reduction, act, norm=False):
        super(ResidualGroup, self).__init__()

        modules_body = [Block(n_feat, n_feat, kernel_size, reduction, bias=True, norm=norm, act=act)
            for _ in range(n_resblocks)]
        modules_body.append(ConvNorm(n_feat, n_feat, kernel_size, stride=1, norm=norm))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


def pixel_shuffle(input, scale_factor):
    batch_size, channels, in_height, in_width = input.size()

    out_channels = int(int(channels / scale_factor) / scale_factor)
    out_height = int(in_height * scale_factor)
    out_width = int(in_width * scale_factor)

    if scale_factor >= 1:
        input_view = input.contiguous().view(batch_size, out_channels, scale_factor, scale_factor, in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    else:
        block_size = int(1 / scale_factor)
        input_view = input.contiguous().view(batch_size, channels, out_height, block_size, out_width, block_size)
        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

    return shuffle_out.view(batch_size, out_channels, out_height, out_width)


class PixelShuffle(nn.Module):
    def __init__(self, scale_factor):
        super(PixelShuffle, self).__init__()
        self.scale_factor = scale_factor
    
    def forward(self, x):
        return pixel_shuffle(x, self.scale_factor)
    def extra_repr(self):
        return 'scale_factor={}'.format(self.scale_factor)


def conv(in_channels, out_channels, kernel_size, 
         stride=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        padding=kernel_size//2,
        stride=1,
        bias=bias,
        groups=groups)


def conv1x1(in_channels, out_channels, stride=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=bias,
        groups=groups)

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def conv5x5(in_channels, out_channels, stride=1, 
            padding=2, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=5,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def conv7x7(in_channels, out_channels, stride=1, 
            padding=3, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=7,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='shuffle'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1)
    elif mode == 'shuffle':
        return nn.Sequential(
            conv3x3(in_channels, 4*out_channels),
            PixelShuffle(2))
    else:
        # out_channels is always going to be the same as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
            conv1x1(in_channels, out_channels))



class Interpolation(nn.Module):
    def __init__(self, n_resgroups, n_resblocks, n_feats, 
                 reduction=16, act=nn.LeakyReLU(0.2, True), norm=False):
        super(Interpolation, self).__init__()

        # define modules: head, body, tail
        self.headConv = conv3x3(n_feats * 2, n_feats)

        modules_body = [
            ResidualGroup(
                RCAB,
                n_resblocks=n_resblocks,
                n_feat=n_feats,
                kernel_size=3,
                reduction=reduction, 
                act=act, 
                norm=norm)
            for _ in range(n_resgroups)]
        self.body = nn.Sequential(*modules_body)

        self.tailConv = conv3x3(n_feats, n_feats)

    def forward(self, x0, x1):
        # Build input tensor
        x = torch.cat([x0, x1], dim=1)
        x = self.headConv(x)

        res = self.body(x)
        res += x

        out = self.tailConv(res)
        return out


class Interpolation_res(nn.Module):
    def __init__(self, n_resgroups, n_resblocks, n_feats,
                 act=nn.LeakyReLU(0.2, True), norm=False):
        super(Interpolation_res, self).__init__()

        # define modules: head, body, tail (reduces concatenated inputs to n_feat)
        self.headConv = conv3x3(n_feats * 2, n_feats)

        modules_body = [ResidualGroup(ResBlock, n_resblocks=n_resblocks, n_feat=n_feats, kernel_size=3,
                            reduction=0, act=act, norm=norm)
                        for _ in range(n_resgroups)]
        self.body = nn.Sequential(*modules_body)

        self.tailConv = conv3x3(n_feats, n_feats)

    def forward(self, x0, x1):
        # Build input tensor
        x = torch.cat([x0, x1], dim=1)
        x = self.headConv(x)

        res = x
        for m in self.body:
            res = m(res)
        res += x

        x = self.tailConv(res)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_pool = self.avg_pool(x).view(x.size(0), -1)
        channel_att = self.fc(avg_pool).view(x.size(0), x.size(1), 1, 1)
        return x * channel_att

class ResidualBlockWithChannelAttention(nn.Module):
    def __init__(self, in_features=32, out_features=32):
        super(ResidualBlockWithChannelAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2)
        )
        self.channel_attention = ChannelAttention(out_features)

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.channel_attention(x1)
        return x1 + x

class ConvBlockWithChannelAttention(nn.Module):
    def __init__(self, in_features, out_features):
        super(ConvBlockWithChannelAttention, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2)
        )

        self.channel_attention = ChannelAttention(out_features)

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.channel_attention(x1)
        return x1
    
class ResNet(nn.Module):
    def __init__(self, hidden_channels):
        super(ResNet, self).__init__()
        self.conv_block = ConvBlockWithChannelAttention(15, hidden_channels)
        self.res_block1 = ResidualBlockWithChannelAttention(hidden_channels, hidden_channels)
        self.res_block2 = ResidualBlockWithChannelAttention(hidden_channels, hidden_channels)
        self.res_block3 = ResidualBlockWithChannelAttention(hidden_channels, hidden_channels)
        self.res_block4 = ResidualBlockWithChannelAttention(hidden_channels, hidden_channels)
        self.res_block5 = ResidualBlockWithChannelAttention(hidden_channels, hidden_channels)
        self.conv_final = nn.Conv2d(hidden_channels, 3, kernel_size=1, stride=1, padding=0)
        
        # Initialize the weights with Kaiming initialization
        self.apply(self._init_weights)

        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total ResNet Parameters: {total_trainable_params}")

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        # ConvBlock
        x = self.conv_block(x)

        # Residual Block 1
        x = self.res_block1(x)

        # Residual Block 2
        x = self.res_block2(x)

        # Residual Block 3
        x = self.res_block3(x)

        # Residual Block 4
        x = self.res_block4(x)

        # Residual Block 5
        x = self.res_block5(x)

        # Final Convolutional Layer
        x = self.conv_final(x)

        return x
    
class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x