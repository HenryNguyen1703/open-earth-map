import torch.nn as nn

from src.models.blocks import Bottleneck, DecoderBlock, EncoderBlock


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=9, base_channels=32):
        super().__init__()

        c = base_channels

        self.encoder1 = EncoderBlock(in_channels, c)
        self.encoder2 = EncoderBlock(c, c * 2)
        self.encoder3 = EncoderBlock(c * 2, c * 4)
        self.encoder4 = EncoderBlock(c * 4, c * 8)
        self.encoder5 = EncoderBlock(c * 8, c * 16)

        self.bottleneck = Bottleneck(c * 16, c * 32)

        self.decoder1 = DecoderBlock(c * 32, c * 16)
        self.decoder2 = DecoderBlock(c * 16, c * 8)
        self.decoder3 = DecoderBlock(c * 8, c * 4)
        self.decoder4 = DecoderBlock(c * 4, c * 2)
        self.decoder5 = DecoderBlock(c * 2, c)

        self.output_conv = nn.Conv2d(
            c,
            num_classes,
            kernel_size=1,
        )

    def forward(self, x):
        skip1, x = self.encoder1(x)
        skip2, x = self.encoder2(x)
        skip3, x = self.encoder3(x)
        skip4, x = self.encoder4(x)
        skip5, x = self.encoder5(x)

        x = self.bottleneck(x)

        x = self.decoder1(x, skip5)
        x = self.decoder2(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder4(x, skip2)
        x = self.decoder5(x, skip1)

        return self.output_conv(x)
