import torch
import torch.nn as nn
from loguru import logger


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.1, inplace=True) # the paper mentioned leaky relu 
        
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = ConvBlock(1, 64)
        self.enc2 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Middle
        self.middle = ConvBlock(128, 256)
        
        # Decoder
        self.upconv = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.output_layer = nn.Conv2d(64, 19, kernel_size=1)
        
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        middle = self.middle(self.pool(enc2))
        
        upconv = self.upconv(middle)
        dec1 = torch.cat([upconv, enc2], dim=1)
        dec1 = self.dec1(dec1)
        
        upconv2 = self.upconv2(dec1)
        dec2 = torch.cat([upconv2, enc1], dim=1)
        dec2 = self.dec2(dec2)
        
        output = self.output_layer(dec2)

        return output



if __name__ == "__main__":

    model = UNet(in_channels=1, out_channels=19)

    # Test with random input
    input_data = torch.randn(1, 1, 512, 512)  # Batch size of 1, 1 channel, 512x512 input dimensions
    output_data = model(input_data)

    logger.info(output_data.shape)  # Output shape should be torch.Size([1, 19, 512, 512])
