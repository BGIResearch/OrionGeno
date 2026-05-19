import torch
import torch.nn as nn
import torch.nn.functional as F

from .species_conditioning import build_species_condition


class RootMeanSquareBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            variance = x.pow(2).mean(dim=(0, 2), keepdim=False)
            self.running_var.mul_(1 - self.momentum).add_(
                variance.detach() * self.momentum
            )
        else:
            variance = self.running_var
        x = x / torch.sqrt(variance.view(1, -1, 1) + self.eps)
        return self.weight.view(1, -1, 1) * x + self.bias.view(1, -1, 1)


class WeightStandardizedConv1d(nn.Conv1d):
    def forward(self, x):
        weight = self.weight
        mean = weight.mean(dim=(1, 2), keepdim=True)
        variance = weight.var(dim=(1, 2), unbiased=False, keepdim=True)
        weight = (weight - mean) / torch.sqrt(variance + 1e-5)
        return F.conv1d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2 if kernel_size > 1 else 0
        self.normalization = RootMeanSquareBatchNorm1d(in_channels)
        self.activation = nn.GELU()
        self.convolution = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
        )

    def forward(self, x):
        return self.convolution(self.activation(self.normalization(x)))


class DownsampleResidualBlock(nn.Module):
    def __init__(self, input_channels, channel_growth=64):
        super().__init__()
        output_channels = input_channels + channel_growth
        self.main_branch = ConvolutionBlock(input_channels, output_channels)
        self.residual_projection = nn.Conv1d(input_channels, output_channels, kernel_size=1)
        self.output_block = ConvolutionBlock(output_channels, output_channels)

    def forward(self, x):
        output = self.main_branch(x)
        output = output + self.residual_projection(x)
        return self.output_block(output)


class SequenceStemBlock(nn.Module):
    def __init__(self, input_channels=5, output_channels=768):
        super().__init__()
        self.input_projection = WeightStandardizedConv1d(
            input_channels,
            output_channels,
            kernel_size=15,
            padding=7,
        )
        self.residual_block = ConvolutionBlock(
            output_channels,
            output_channels,
            kernel_size=5,
        )

    def forward(self, x):
        output = self.input_projection(x)
        return output + self.residual_block(output)


class SequencePyramidEncoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        settings = getattr(config, "settings", {})
        input_channels = settings.get("input_channels", 5)
        self.output_channels = settings.get(
            "encoder_output_channels",
            config.d_model // 2,
        )
        pooling_kernel_size = settings.get("pooling_kernel_size", 2)
        self.feature_scales = list(settings.get("feature_scales", [1, 2, 4, 8, 16, 32, 64]))
        channel_growth = settings.get("channel_growth", 64)

        self.stages = nn.ModuleList([SequenceStemBlock(input_channels, self.output_channels)])
        stage_channels = self.output_channels
        for _ in range(len(self.feature_scales) - 1):
            self.stages.append(
                DownsampleResidualBlock(stage_channels, channel_growth=channel_growth)
            )
            stage_channels += channel_growth
        self.pool = nn.MaxPool1d(pooling_kernel_size, pooling_kernel_size)

    def forward(self, x, species_embedding=None):
        x = x.permute(0, 2, 1)
        species_condition = build_species_condition(
            species_embedding,
            expected_channels=self.output_channels,
            context="encoder scale-1 features",
        )
        skip_connections = {}
        for stage, scale in zip(self.stages, self.feature_scales):
            x = stage(x)
            if scale == 1 and species_condition is not None:
                x = x + species_condition
            skip_connections[f"scale_{scale}"] = x
            x = self.pool(x)

        return x.permute(0, 2, 1), skip_connections


class UpsampleResidualBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, residual_scale_init=0.9):
        super().__init__()
        self.main_branch = ConvolutionBlock(in_channels, skip_channels, kernel_size=5)
        self.input_projection = nn.Conv1d(in_channels, skip_channels, kernel_size=1)
        self.skip_branch = ConvolutionBlock(skip_channels, skip_channels, kernel_size=1)
        self.output_block = ConvolutionBlock(skip_channels, skip_channels, kernel_size=5)
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale_init))

    def forward(self, x, skip_tensor):
        output = self.main_branch(x)
        output = output + self.input_projection(x)
        output = output.repeat_interleave(2, dim=2)
        output = output * self.residual_scale
        output = output + self.skip_branch(skip_tensor)
        return output + self.output_block(output)


class SequencePyramidDecoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        settings = getattr(config, "settings", {})
        decoder_channel_sizes = settings.get(
            "decoder_channel_sizes",
            [768, 704, 640, 576, 512, 448, 384],
        )
        feature_scales = list(settings.get("feature_scales", [1, 2, 4, 8, 16, 32, 64]))
        self.skip_scales = list(reversed(feature_scales))
        self.stages = nn.ModuleList(
            [
                UpsampleResidualBlock(in_channels, skip_channels)
                for in_channels, skip_channels in zip(
                    decoder_channel_sizes[:-1],
                    decoder_channel_sizes[1:],
                )
            ]
        )
        if len(self.skip_scales) != len(self.stages):
            raise ValueError(
                "decoder_channel_sizes must contain exactly one more entry than "
                "feature_scales for the configured OrionGeno decoder."
            )

    def forward(self, x, skip_connections, species_embedding=None):
        if x.dim() == 3 and x.shape[1] != skip_connections["scale_1"].shape[1]:
            x = x.permute(0, 2, 1)

        output_channels = self.stages[-1].output_block.convolution.out_channels
        species_condition = build_species_condition(
            species_embedding,
            expected_channels=output_channels,
            context="decoder output features",
        )

        for index, scale in enumerate(self.skip_scales):
            x = self.stages[index](x, skip_connections[f"scale_{scale}"])
        if species_condition is not None:
            x = x + species_condition
        return x.permute(0, 2, 1)


def encode_tokens_as_one_hot(
    input_ids,
    vocab_size=11,
    output_size=9,
    pad_id=9,
    mask_id=10,
    dtype=None,
):
    dtype = dtype or torch.float32
    full_one_hot = F.one_hot(input_ids, num_classes=vocab_size).to(dtype=dtype)
    sequence_channels = full_one_hot[:, :, :output_size]
    pad_mask = input_ids == pad_id
    mask_mask = input_ids == mask_id
    sequence_channels[pad_mask] = 0.0
    sequence_channels[mask_mask] = 1.0 / output_size
    return sequence_channels
