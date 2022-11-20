import torch
import torch.nn as nn
import torch.nn.functional as F



class UNet2D(nn.Module):
    # Implemented from https://arxiv.org/pdf/1505.04597.pdf
    def __init__(self):
        super(UNet2D, self).__init__()
        kernel_size_ = 2
        stride_ = 2

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = self.__double_conv2d(1, 64)
        self.down_conv_2 = self.__double_conv2d(64, 128)
        self.down_conv_3 = self.__double_conv2d(128, 256)
        self.down_conv_4 = self.__double_conv2d(256, 512)
        self.down_conv_5 = self.__double_conv2d(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024,
                                             out_channels=512,
                                             kernel_size=kernel_size_,
                                             stride=stride_
                                             )
        self.up_conv_1 = self.__double_conv2d(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(in_channels=512,
                                             out_channels=256,
                                             kernel_size=kernel_size_,
                                             stride=stride_
                                             )
        self.up_conv_2 = self.__double_conv2d(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(in_channels=256,
                                             out_channels=128,
                                             kernel_size=kernel_size_,
                                             stride=stride_
                                             )
        self.up_conv_3 = self.__double_conv2d(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(in_channels=128,
                                             out_channels=64,
                                             kernel_size=kernel_size_,
                                             stride=stride_
                                             )
        self.up_conv_4 = self.__double_conv2d(128, 64)

        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=8,
            kernel_size=(1, 1)
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
        # y = self.__crop_img2d(x7, x)
        x = self.up_conv_1(torch.cat([x7, x], 1))

        x = self.up_trans_2(x)
        # y = self.__crop_img2d(x5, x)
        x = self.up_conv_2(torch.cat([x5, x], 1))

        x = self.up_trans_3(x)
        # y = self.__crop_img2d(x3, x)
        x = self.up_conv_3(torch.cat([x3, x], 1))

        x = self.up_trans_4(x)
        # y = self.__crop_img2d(x1, x)
        x = self.up_conv_4(torch.cat([x1, x], 1))

        x = self.out(x)
        # x = F.softmax(x, dim=1) #Don't use if you use crossentropy loss because it has been already implemented in it.

        return x

    @staticmethod
    def __double_conv2d(in_c, out_c, kernel_size_=(3, 3)):
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size_, padding="same", padding_mode="zeros"),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=kernel_size_, padding="same", padding_mode="zeros"),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

        return conv

    @staticmethod
    def __crop_img2d(source_tensor, target_tensor):
        target_size = target_tensor.size()[2]
        tensor_size = source_tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2

        return source_tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]


class UNet2Dv2(nn.Module):
    # Implemented from https://arxiv.org/pdf/1505.04597.pdf and modified.
    def __init__(self):
        super(UNet2Dv2, self).__init__()
        kernel_size_ = 2
        stride_ = 2

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = self.__double_conv2d(1, 32)
        self.down_conv_2 = self.__double_conv2d(32, 64)
        self.down_conv_3 = self.__double_conv2d(64, 128)
        self.down_conv_4 = self.__double_conv2d(128, 256)

        self.up_trans_1 = nn.ConvTranspose2d(in_channels=256,
                                             out_channels=128,
                                             kernel_size=kernel_size_,
                                             stride=stride_
                                             )
        self.up_conv_1 = self.__double_conv2d(256, 128)

        self.up_trans_2 = nn.ConvTranspose2d(in_channels=128,
                                             out_channels=64,
                                             kernel_size=kernel_size_,
                                             stride=stride_
                                             )
        self.up_conv_2 = self.__double_conv2d(128, 64)

        self.up_trans_3 = nn.ConvTranspose2d(in_channels=64,
                                             out_channels=32,
                                             kernel_size=kernel_size_,
                                             stride=stride_
                                             )
        self.up_conv_3 = self.__double_conv2d(64, 32)

        self.out = nn.Conv2d(
            in_channels=32,
            out_channels=8,
            kernel_size=(1, 1)
        )

    def forward(self, image):
        # encoder
        x1 = self.down_conv_1(image)  #
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)  #
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)  #
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)

        # decoder
        x = self.up_trans_1(x7)
        x = self.up_conv_1(torch.cat([x5, x], 1))
        x = self.up_trans_2(x)
        x = self.up_conv_2(torch.cat([x3, x], 1))
        x = self.up_trans_3(x)
        x = self.up_conv_3(torch.cat([x1, x], 1))
        x = self.out(x)
        # x = F.softmax(x, dim=1) Don't use if you use crossentropy loss because it has been already implemented in it.

        return x

    @staticmethod
    def __double_conv2d(in_c, out_c, kernel_size_=(3, 3)):
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size_, padding="same", padding_mode="zeros"),
            nn.BatchNorm2d(out_c),
            # nn.GroupNorm(4, out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=kernel_size_, padding="same", padding_mode="zeros"),
            nn.BatchNorm2d(out_c),
            # nn.GroupNorm(4, out_c),
            nn.ReLU(inplace=True),
        )

        return conv