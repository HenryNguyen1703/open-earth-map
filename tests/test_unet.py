import torch

from src.models.blocks import ConvBlock


def test_conv_block_preserves_spatial_size_and_changes_channels():
    block = ConvBlock(
        in_channels=3,
        out_channels=32,
    )

    x = torch.randn(2, 3, 64, 64)

    output = block(x)

    assert output.shape == (2, 32, 64, 64)


def test_encoder_block_returns_skip_features_and_downsampled_output():
    from src.models.blocks import EncoderBlock

    block = EncoderBlock(
        in_channels=3,
        out_channels=32,
    )

    x = torch.randn(2, 3, 64, 64)

    features, pooled = block(x)

    assert features.shape == (2, 32, 64, 64)
    assert pooled.shape == (2, 32, 32, 32)


def test_bottleneck_changes_channels_and_preserves_spatial_size():
    from src.models.blocks import Bottleneck

    block = Bottleneck(
        in_channels=512,
        out_channels=1024,
    )

    x = torch.randn(2, 512, 4, 4)

    output = block(x)

    assert output.shape == (2, 1024, 4, 4)


def test_decoder_block_upsamples_and_merges_skip_connection():
    from src.models.blocks import DecoderBlock

    block = DecoderBlock(
        in_channels=1024,
        out_channels=512,
    )

    x = torch.randn(2, 1024, 4, 4)
    skip = torch.randn(2, 512, 8, 8)

    output = block(x, skip)

    assert output.shape == (2, 512, 8, 8)


def test_decoder_block_uses_upsample_instead_of_transposed_convolution():
    import torch.nn as nn
    from src.models.blocks import DecoderBlock

    block = DecoderBlock(
        in_channels=1024,
        out_channels=512,
    )

    assert isinstance(block.up, nn.Upsample)


def test_unet_returns_logits_with_same_spatial_size_and_num_classes():
    from src.models.unet import UNet

    model = UNet(
        in_channels=3,
        num_classes=9,
        base_channels=32,
    )

    x = torch.randn(2, 3, 64, 64)

    logits = model(x)

    assert logits.shape == (2, 9, 64, 64)
