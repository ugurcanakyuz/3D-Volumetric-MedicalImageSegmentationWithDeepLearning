import torch
import torch.nn as nn
import torch.nn.functional as F



class UNet3Dv2(nn.Module):
    # Implemented from https://arxiv.org/pdf/1606.06650.pdf.
    def __init__(self, n_classes=8):
        super(UNet3Dv2, self).__init__()
        up_kernel_size = (2, 2, 2)
        up_stride = (2, 2, 2)

        self.max_pool_2x2x2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.down_conv_input = nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding="same", padding_mode="zeros")
        self.down_conv1 = self.__double_down_conv(8, 64)
        self.down_conv2 = self.__double_down_conv(64, 128)
        self.down_conv3 = self.__double_down_conv(128, 256)
        self.down_conv4 = self.__double_down_conv(256, 512)

        self.up_trans1 = nn.ConvTranspose3d(512, 512, kernel_size=up_kernel_size, stride=up_stride)
        self.up_conv1 = self.__double_up_conv(in_channels=1024, out_channels=512)

        self.up_trans2 = nn.ConvTranspose3d(512, 512, kernel_size=up_kernel_size, stride=up_stride)
        self.up_conv2 = self.__double_up_conv(in_channels=768, out_channels=256)

        self.up_trans3 = nn.ConvTranspose3d(256, 256, kernel_size=up_kernel_size, stride=up_stride)
        self.up_conv3 = self.__double_up_conv(in_channels=384, out_channels=128)

        self.up_trans4 = nn.ConvTranspose3d(128, 128, kernel_size=up_kernel_size, stride=up_stride)
        self.up_conv4 = self.__double_up_conv(in_channels=192, out_channels=64)

        self.out = nn.Conv3d(in_channels=64, out_channels=n_classes, kernel_size=(1, 1, 1))

    def forward(self, image):
        # bs, c, d, h, w
        # encoder
        xi = self.down_conv_input(image)
        x1 = self.down_conv1(xi)
        x2 = self.max_pool_2x2x2(x1)
        x3 = self.down_conv2(x2)
        x4 = self.max_pool_2x2x2(x3)
        x5 = self.down_conv3(x4)
        x6 = self.max_pool_2x2x2(x5)
        x7 = self.down_conv4(x6)
        x8 = self.max_pool_2x2x2(x7)

        # decoder
        x = self.up_trans1(x8)
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
    def __double_down_conv(in_channels, out_channels, kernel_size_=(3, 3, 3)):
        conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, kernel_size=kernel_size_, padding="same", padding_mode="zeros"),
            # nn.BatchNorm3d(out_channels // 2),
            nn.GroupNorm(8, out_channels // 2),
            nn.ReLU(inplace=True),
            # nn.Mish(inplace=True),
            nn.Conv3d(out_channels // 2, out_channels, kernel_size=kernel_size_, padding="same", padding_mode="zeros"),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
            # nn.Mish(inplace=True)
        )

        return conv

    @staticmethod
    def __double_up_conv(in_channels, out_channels, kernel_size_=(3, 3, 3)):
        conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=kernel_size_, padding="same", padding_mode="zeros"),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(8, in_channels // 2),
            nn.ReLU(inplace=True),
            # nn.Mish(inplace=True),
            nn.Conv3d(in_channels // 2, out_channels, kernel_size=kernel_size_, padding="same", padding_mode="zeros"),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
            # nn.Mish(inplace=True)
        )

        return conv
