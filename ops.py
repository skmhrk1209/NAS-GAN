import torch
from torch import nn


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride,
                 affine, preactivation=True, **kwargs):
        super().__init__()
        self.module = nn.Sequential(
            nn.ReLU() if preactivation else nn.Identity(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
                affine=affine
            )
        )

    def forward(self, input):
        return self.module(input)


class ConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride,
                 affine, preactivation=True, **kwargs):
        super().__init__()
        self.module = nn.Sequential(
            nn.ReLU() if preactivation else nn.Identity(),
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False,
                output_padding=0 if stride == 1 else 1
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
                affine=affine
            )
        )

    def forward(self, input):
        return self.module(input)


class DilatedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride,
                 dilation, affine, preactivation=True, **kwargs):
        super().__init__()
        self.module = nn.Sequential(
            nn.ReLU() if preactivation else nn.Identity(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                groups=in_channels,
                bias=False
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
                affine=affine
            )
        )

    def forward(self, input):
        return self.module(input)


class DilatedConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride,
                 dilation, affine, preactivation=True, **kwargs):
        super().__init__()
        self.module = nn.Sequential(
            nn.ReLU() if preactivation else nn.Identity(),
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                groups=in_channels,
                bias=False,
                output_padding=0 if stride == 1 else 1
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
                affine=affine
            )
        )

    def forward(self, input):
        return self.module(input)


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride,
                 affine, preactivation=True, **kwargs):
        super().__init__()
        self.module = nn.Sequential(
            nn.ReLU() if preactivation else nn.Identity(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                groups=in_channels,
                bias=False
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=in_channels,
                affine=affine
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=1,
                groups=in_channels,
                bias=False
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
                affine=affine
            )
        )

    def forward(self, input):
        return self.module(input)


class SeparableConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride,
                 affine, preactivation=True, **kwargs):
        super().__init__()
        self.module = nn.Sequential(
            nn.ReLU() if preactivation else nn.Identity(),
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                groups=in_channels,
                bias=False,
                output_padding=0 if stride == 1 else 1
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=in_channels,
                affine=affine
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=1,
                groups=in_channels,
                bias=False
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
                affine=affine
            )
        )

    def forward(self, input):
        return self.module(input)


class Identity(nn.Module):

    def __init__(self, in_channels, out_channels, stride, affine, preactivation=True, **kwargs):
        super().__init__()
        self.module = nn.Identity() if stride == 1 and in_channels == out_channels else Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=stride,
            affine=affine,
            preactivation=preactivation
        )

    def forward(self, input):
        return self.module(input)


class IdentityTranspose(nn.Module):

    def __init__(self, in_channels, out_channels, stride, affine, preactivation=True, **kwargs):
        super().__init__()
        self.module = nn.Identity() if stride == 1 and in_channels == out_channels else ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=stride,
            affine=affine,
            preactivation=preactivation
        )

    def forward(self, input):
        return self.module(input)


class Zero(nn.Module):

    def __init__(self, stride, **kwargs):
        super().__init__()
        self.stride = stride

    def forward(self, input):
        shape = [input.shape[-2] // self.stride, input.shape[-1] // self.stride]
        return input.new_zeros(*input.shape[:-2], *shape)


class ZeroTranspose(nn.Module):

    def __init__(self, stride, **kwargs):
        super().__init__()
        self.stride = stride

    def forward(self, input):
        shape = [input.shape[-2] * self.stride, input.shape[-1] * self.stride]
        return input.new_zeros(*input.shape[:-2], *shape)


class ScheduledGumbelSoftmax(nn.Module):

    def __init__(self, temperature_fn):
        super().__init__()
        self.temperature_fn = temperature_fn
        self.epoch = 0

    def forward(self, input, dim=-1, hard=False):
        temperature = self.temperature_fn(self.epoch)
        uniform = torch.rand_like(input)
        gumbel = -torch.log(-torch.log(uniform))
        soft = nn.functional.softmax((input + gumbel) / temperature, dim=dim)
        if hard:
            index = torch.argmax(soft, dim=dim, keepdim=True)
            hard = torch.scatter(torch.zeros_like(soft), dim=dim, index=index, value=1)
            return hard + soft - soft.detach()
        else:
            return soft

    def step(self, epoch=None):
        self.epoch = epoch or self.epoch + 1


class ScheduledDropPath(nn.Module):

    def __init__(self, drop_prob_fn):
        super().__init__()
        self.drop_prob_fn = drop_prob_fn
        self.epoch = 0

    def forward(self, input):
        drop_prob = self.drop_prob_fn(self.epoch)
        if self.training and drop_prob > 0:
            keep_prob = 1 - drop_prob
            mask = input.new_full((input.size(0), 1, 1, 1), keep_prob)
            mask = torch.bernoulli(mask)
            input = input * mask
            input = input / keep_prob
        return input

    def step(self, epoch=None):
        self.epoch = epoch or self.epoch + 1


class Cutout(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, input):
        y_min = torch.randint(input.size(-2) - self.size[-2], (1,))
        x_min = torch.randint(input.size(-1) - self.size[-1], (1,))
        y_max = y_min + self.size[-2]
        x_max = x_min + self.size[-1]
        input[..., y_min:y_max, x_min:x_max] = 0
        return input


class CrossEntropyLoss(nn.Module):

    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, input, target, dim=-1):
        log_prob = nn.functional.log_softmax(input, dim=dim)
        target = torch.unsqueeze(target, dim=dim)
        target = torch.scatter(torch.zeros_like(log_prob), dim=dim, index=target, value=1)
        target = (1 - self.smoothing) * target + self.smoothing / (target.size(1) - 1) * (1 - target)
        loss = -torch.mean(torch.sum(target * log_prob, dim=dim))
        return loss
