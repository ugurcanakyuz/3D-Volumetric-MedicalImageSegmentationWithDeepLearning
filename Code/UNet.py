import torch
import torch.nn as nn

class UNet3D(nn.Module):
    # Implemented from https://arxiv.org/pdf/1606.06650.pdf.
    def __init__(self):
        super(UNet3D, self).__init__()

        self.max_pool_2x2x2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.down_conv1 = self.__double_conv(3, 32)
        self.down_conv2 = self.__double_conv(64, 128)
        self.down_conv3 = self.__double_conv(128, 256)
        self.down_conv4 = self.__double_conv(256, 512)

    def forward(self, image):
        # bs, c, h, w
        # encoder
        x1 = self.down_conv1(image)
        x2 = self.max_pool_2x2x2(x1)
        x3 = self.down_conv2(x2)
        x4 = self.max_pool_2x2x2(x3)
        x5 = self.down_conv3(x4)
        x6 = self.max_pool_2x2x2(x5)
        x7 = self.down_conv3(x6)
        x8 = self.max_pool_2x2x2(x7)
        print(x1.size())
        print(x3.size())
        print(x5.size())
        print(x7.size())
        print(x8.size())

        return x8


    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels*2, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        return conv

if __name__ == "__main__":
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = torch.rand((1, 3, 256, 256, 256))
    model = UNet3D()
    print(model(image))