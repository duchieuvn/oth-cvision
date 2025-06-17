import torch.nn as nn
import torch
from collections import OrderedDict
from monai.networks.nets import UNet as MonaiUnet
from monai.networks.nets import BasicUNetPlusPlus
from collections.abc import Sequence
from monai.networks.layers.factories import Conv
from monai.networks.nets.basic_unet import Down, TwoConv, UpSample
from monai.utils import ensure_tuple_rep

class UNetConcat(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNetConcat, self).__init__()

        features = init_features
        self.encoder1 = UNetConcat._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNetConcat._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNetConcat._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNetConcat._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNetConcat._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNetConcat._block((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNetConcat._block((features * 4) * 2, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNetConcat._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNetConcat._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )




class UpSum(nn.Module):
    def __init__(self, spatial_dims, in_channels, skip_channels, out_channels,
                 act, norm, bias, dropout, upsample_mode, halves=True):
        super().__init__()
        self.halves = halves
        self.up = UpSample(spatial_dims, in_channels, out_channels if halves else in_channels,
                           2, mode=upsample_mode)

        self.align_skip = nn.Conv3d(skip_channels, out_channels, kernel_size=1) if spatial_dims == 3 else \
                          nn.Conv2d(skip_channels, out_channels, kernel_size=1)

        self.conv = TwoConv(spatial_dims, out_channels, out_channels, act, norm, bias, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.align_skip(skip)
        x = x + skip
        return self.conv(x)

class BasicUNetPlusPlusSum(nn.Module):
    def __init__(self,
                 spatial_dims: int = 3,
                 in_channels: int = 1,
                 out_channels: int = 2,
                 features: Sequence[int] = (32, 32, 64, 128, 256, 32),
                 deep_supervision: bool = False,
                 act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                 norm: str | tuple = ("instance", {"affine": True}),
                 bias: bool = True,
                 dropout: float | tuple = 0.0,
                 upsample: str = "deconv"):
        super().__init__()
        self.deep_supervision = deep_supervision
        fea = ensure_tuple_rep(features, 6)

        self.conv_0_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
        self.conv_1_0 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.conv_2_0 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.conv_3_0 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.conv_4_0 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        self.up_0_1 = UpSum(spatial_dims, fea[1], fea[0], fea[0], act, norm, bias, dropout, upsample, halves=False)
        self.up_1_1 = UpSum(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.up_2_1 = UpSum(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.up_3_1 = UpSum(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)

        self.up_0_2 = UpSum(spatial_dims, fea[1], fea[0], fea[0], act, norm, bias, dropout, upsample, halves=False)
        self.up_1_2 = UpSum(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.up_2_2 = UpSum(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)

        self.up_0_3 = UpSum(spatial_dims, fea[1], fea[0], fea[0], act, norm, bias, dropout, upsample, halves=False)
        self.up_1_3 = UpSum(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)

        self.up_0_4 = UpSum(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv_0_1 = Conv["conv", spatial_dims](fea[0], out_channels, kernel_size=1)
        self.final_conv_0_2 = Conv["conv", spatial_dims](fea[0], out_channels, kernel_size=1)
        self.final_conv_0_3 = Conv["conv", spatial_dims](fea[0], out_channels, kernel_size=1)
        self.final_conv_0_4 = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

    def forward(self, x):
        x_0_0 = self.conv_0_0(x)
        x_1_0 = self.conv_1_0(x_0_0)
        x_0_1 = self.up_0_1(x_1_0, x_0_0)

        x_2_0 = self.conv_2_0(x_1_0)
        x_1_1 = self.up_1_1(x_2_0, x_1_0)
        x_0_2 = self.up_0_2(x_1_1, x_0_0 + x_0_1)

        x_3_0 = self.conv_3_0(x_2_0)
        x_2_1 = self.up_2_1(x_3_0, x_2_0)
        x_1_2 = self.up_1_2(x_2_1, x_1_0 + x_1_1)
        x_0_3 = self.up_0_3(x_1_2, x_0_0 + x_0_1 + x_0_2)

        x_4_0 = self.conv_4_0(x_3_0)
        x_3_1 = self.up_3_1(x_4_0, x_3_0)
        x_2_2 = self.up_2_2(x_3_1, x_2_0 + x_2_1)
        x_1_3 = self.up_1_3(x_2_2, x_1_0 + x_1_1 + x_1_2)
        x_0_4 = self.up_0_4(x_1_3, x_0_0 + x_0_1 + x_0_2 + x_0_3)

        out_0_1 = self.final_conv_0_1(x_0_1)
        out_0_2 = self.final_conv_0_2(x_0_2)
        out_0_3 = self.final_conv_0_3(x_0_3)
        out_0_4 = self.final_conv_0_4(x_0_4)

        return [out_0_1, out_0_2, out_0_3, out_0_4] if self.deep_supervision else [out_0_4]
