import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_=3,
                 padding_="same", padding_mode_="zeros"):
        super(DilatedBlock, self).__init__()



        self.conv1 = nn.Conv3d(in_channels, out_channels // 2,
                               kernel_size=kernel_size_, padding=padding_, padding_mode=padding_mode_, dilation=1)
        self.gn1 = nn.GroupNorm(8, out_channels // 2)
        self.conv2 = nn.Conv3d(out_channels // 2, out_channels // 4,
                               kernel_size=kernel_size_, padding="same", padding_mode="zeros", dilation=3)
        self.gn2 = nn.GroupNorm(8, out_channels // 4)
        self.conv3 = nn.Conv3d(out_channels // 4, out_channels // 8,
                               kernel_size=kernel_size_, padding="same", padding_mode="zeros", dilation=6)
        self.gn3 = nn.GroupNorm(4, out_channels // 8)
        self.conv4 = nn.Conv3d(out_channels // 8, out_channels // 16,
                               kernel_size=kernel_size_, padding="same", padding_mode="zeros", dilation=9)
        self.gn4 = nn.GroupNorm(4, out_channels // 16)
        self.conv5 = nn.Conv3d(out_channels // 16, out_channels // 16,
                               kernel_size=kernel_size_, padding="same", padding_mode="zeros", dilation=12)

    def forward(self, data):
        out1 = self.conv1(data)
        x = F.relu(self.gn1(out1))
        out2 = self.conv2(x)
        x = F.relu(self.gn2(out2))
        out3 = self.conv3(x)
        x = F.relu(self.gn3(out3))
        out4 = self.conv4(x)
        x = F.relu(self.gn4(out4))
        out5 = F.relu(self.conv5(x))

        out_map = torch.concat([out1, out2, out3, out4, out5], dim=1)

        return out_map

class SDUNet3D(nn.Module):
    # Implemented from https://arxiv.org/pdf/1606.06650.pdf.
    def __init__(self, n_classes=8):
        super(SDUNet3D, self).__init__()
        up_kernel_size = (2, 2, 2)
        up_stride = (2, 2, 2)

        self.max_pool_2x2x2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.down_conv1 = self.__SDBlock(1, 64)
        self.down_conv2 = self.__SDBlock(64, 128)
        self.down_conv3 = self.__SDBlock(128, 256)
        self.down_conv4 = self.__SDBlock(256, 512)
        self.down_conv5 = self.__SDBlock(512, 512)

        self.up_trans1 = nn.ConvTranspose3d(512, 512, kernel_size=up_kernel_size, stride=up_stride)
        self.up_conv1 = self.__SDBlock(in_channels=1024, out_channels=256)

        self.up_trans2 = nn.ConvTranspose3d(256, 256, kernel_size=up_kernel_size, stride=up_stride)
        self.up_conv2 = self.__SDBlock(in_channels=512, out_channels=128)

        self.up_trans3 = nn.ConvTranspose3d(128, 128, kernel_size=up_kernel_size, stride=up_stride)
        self.up_conv3 = self.__SDBlock(in_channels=256, out_channels=64)

        self.up_trans4 = nn.ConvTranspose3d(64, 64, kernel_size=up_kernel_size, stride=up_stride)
        self.up_conv4 = self.__SDBlock(in_channels=128, out_channels=64)

        self.out = nn.Conv3d(in_channels=64, out_channels=n_classes, kernel_size=(1, 1, 1))

    def forward(self, data):
        # bs, c, d, h, w
        # encoder
        x1 = self.down_conv1(data)
        x2 = self.max_pool_2x2x2(x1)
        x3 = self.down_conv2(x2)
        x4 = self.max_pool_2x2x2(x3)
        x5 = self.down_conv3(x4)
        x6 = self.max_pool_2x2x2(x5)
        x7 = self.down_conv4(x6)
        x8 = self.max_pool_2x2x2(x7)
        x9 = self.down_conv5(x8)

        # decoder
        x = self.up_trans1(x9)
        x = torch.cat([x7, x], 1)
        x = self.up_conv1(x)
        x = self.up_trans2(x)
        x = torch.cat([x5, x], 1)
        x = self.up_conv2(x)
        x = self.up_trans3(x)
        x = torch.cat([x3, x], 1)
        x = self.up_conv3(x)
        x = self.up_trans4(x)
        x = torch.cat([x1, x], 1)
        x = self.up_conv4(x)
        x = self.out(x)
        # x = F.softmax(x, dim=1)

        return x

    @staticmethod
    def __SDBlock(in_channels, out_channels):
        conv = nn.Sequential(
            DilatedBlock(in_channels, out_channels),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

        return conv