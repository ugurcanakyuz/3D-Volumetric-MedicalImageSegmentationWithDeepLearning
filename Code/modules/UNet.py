import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet2D(nn.Module):
    # Implemented from https://arxiv.org/pdf/1505.04597.pdf
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = self.__double_conv2d(1, 64)
        self.down_conv_2 = self.__double_conv2d(64, 128)
        self.down_conv_3 = self.__double_conv2d(128, 256)
        self.down_conv_4 = self.__double_conv2d(256, 512)
        self.down_conv_5 = self.__double_conv2d(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024,
                                             out_channels=512,
                                             kernel_size=2,
                                             stride=2
                                             )
        self.up_conv_1 = self.__double_conv2d(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(in_channels=512,
                                             out_channels=256,
                                             kernel_size=2,
                                             stride=2
                                             )
        self.up_conv_2 = self.__double_conv2d(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(in_channels=256,
                                             out_channels=128,
                                             kernel_size=2,
                                             stride=2
                                             )
        self.up_conv_3 = self.__double_conv2d(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(in_channels=128,
                                             out_channels=64,
                                             kernel_size=2,
                                             stride=2
                                             )
        self.up_conv_4 = self.__double_conv2d(128, 64)

        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=8,
            kernel_size=1
        )

    def forward(self, image):
        # encoder
        x1 = self.down_conv_1(image)  #
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)  #
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)  #
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)  #
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)

        # decoder
        x = self.up_trans_1(x9)
        y = self.__crop_img2d(x7, x)
        x = self.up_conv_1(torch.cat([y, x], 1))

        x = self.up_trans_2(x)
        y = self.__crop_img2d(x5, x)
        x = self.up_conv_2(torch.cat([y, x], 1))

        x = self.up_trans_3(x)
        y = self.__crop_img2d(x3, x)
        x = self.up_conv_3(torch.cat([y, x], 1))

        x = self.up_trans_4(x)
        y = self.__crop_img2d(x1, x)
        x = self.up_conv_4(torch.cat([y, x], 1))

        x = self.out(x)

        return x

    def __double_conv2d(in_c, out_c):
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        return conv

    def __crop_img2d(tensor, target_tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2

        return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]


class UNet3D(nn.Module):
    # Implemented from https://arxiv.org/pdf/1606.06650.pdf.
    def __init__(self, n_classes=8):
        super(UNet3D, self).__init__()
        self.pad = (1, 1, 1, 1, 1, 1)

        self.max_pool_2x2x2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.down_conv1 = self.__double_down_conv(1, 64)
        self.down_conv2 = self.__double_down_conv(64, 128)
        self.down_conv3 = self.__double_down_conv(128, 256)
        self.down_conv4 = self.__double_down_conv(256, 512)

        self.up_trans1 = nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2)
        self.up_conv1 = self.__conv3d(in_channels=768, out_channels=256)
        self.up_conv2 = self.__conv3d(in_channels=256, out_channels=256)

        self.up_trans2 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.up_conv3 = self.__conv3d(in_channels=384, out_channels=128)
        self.up_conv4 = self.__conv3d(in_channels=128, out_channels=128)

        self.up_trans3 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.up_conv5 = self.__conv3d(in_channels=192, out_channels=64)
        self.up_conv6 = self.__conv3d(in_channels=64, out_channels=64)
        self.out = nn.Conv3d(in_channels=64, out_channels=n_classes, kernel_size=1)

    def forward(self, image):
        # bs, c, d, h, w
        # encoder
        x1 = self.down_conv1(image)
        x2 = self.max_pool_2x2x2(x1)
        x3 = self.down_conv2(x2)
        x4 = self.max_pool_2x2x2(x3)
        x5 = self.down_conv3(x4)
        x6 = self.max_pool_2x2x2(x5)
        x7 = self.down_conv4(x6)

        # decoder
        x = self.up_trans1(x7)
        x = F.relu(self.up_conv1(torch.cat([x5, x], 1)))
        x = F.relu(self.up_conv2(x))
        x = self.up_trans2(x)
        x = F.relu(self.up_conv3(torch.cat([x3, x], 1)))
        x = F.relu(self.up_conv4(x))
        x = self.up_trans3(x)
        x = F.relu(self.up_conv5(torch.cat([x1, x], 1)))
        x = F.relu(self.up_conv6(x))
        x = F.softmax(self.out(x), dim=2) #ToDo: Be sure if the dim parameter is correct.

        return x

    def __double_down_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, kernel_size=3, padding="same", padding_mode="zeros"),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels // 2, out_channels, kernel_size=3, padding="same", padding_mode="zeros"),
            nn.ReLU(inplace=True)
        )

        return conv

    def __conv3d(self, in_channels, out_channels):
        return nn.Conv3d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=3,
                         padding="same",
                         padding_mode="zeros")

    # def __crop_img3d(self, tensor, target_tensor):
    #    target_size = target_tensor.size()[2]
    #    tensor_size = tensor.size()[2]
    #    delta = tensor_size - target_size
#
#    if delta%2==0:
#        delta = delta//2
#        return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta, delta:tensor_size-delta]
#    else:
#        delta = delta//2
#        return tensor[:, :, delta:tensor_size-delta-1, delta:tensor_size-delta-1, delta:tensor_size-delta-1]
