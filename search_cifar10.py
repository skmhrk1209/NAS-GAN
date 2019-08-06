from torch import distributed
from torch import optim
from torch import utils
from torch import cuda
from torch import backends
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torchvision import ops
from models import *
from ops import *
from datasets import *
from distributed import *
from transforms import *
from training import *
from utils import *
import numpy as np
import collections
import functools
import itertools
import argparse
import shutil
import json
import os


def main(args):

    init_process_group(backend='nccl')

    with open(args.config) as file:
        config = json.load(file)
    config.update(vars(args))
    config = apply_dict(Dict, config)

    backends.cudnn.benchmark = True
    backends.cudnn.fastest = True

    global_rank = distributed.get_rank()
    local_rank = global_rank % cuda.device_count()

    np.random.seed(global_rank)
    torch.manual_seed(global_rank)
    cuda.manual_seed(global_rank)
    cuda.set_device(local_rank)

    train_dataset = datasets.CIFAR10(
        root=config.train_root,
        train=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.49139968, 0.48215827, 0.44653124),
                std=(0.24703233, 0.24348505, 0.26158768)
            )
        ]),
        download=True
    )
    train_train_dataset, train_val_dataset = utils.data.random_split(
        dataset=train_dataset,
        lengths=[int(len(train_dataset) * config.split_ratio), int(len(train_dataset) * (1 - config.split_ratio))]
    )
    val_dataset = datasets.CIFAR10(
        root=config.val_root,
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.49139968, 0.48215827, 0.44653124),
                std=(0.24703233, 0.24348505, 0.26158768)
            )
        ]),
        download=True
    )

    train_train_sampler = utils.data.distributed.DistributedSampler(train_train_dataset)
    train_val_sampler = utils.data.distributed.DistributedSampler(train_val_dataset)
    val_sampler = utils.data.distributed.DistributedSampler(val_dataset)

    train_train_data_loaders = utils.data.DataLoader(
        dataset=train_train_dataset,
        batch_size=config.local_batch_size,
        sampler=train_train_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )
    train_val_data_loaders = utils.data.DataLoader(
        dataset=train_val_dataset,
        batch_size=config.local_batch_size,
        sampler=train_val_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_data_loader = utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config.local_batch_size,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )

    generator = DARTSGenerator(
        latent_size=128,
        min_resolution=4,
        out_channels=3,
        operations=dict(
            sep_conv_3x3=functools.partial(SeparableConvTranspose2d, kernel_size=3, padding=1),
            sep_conv_5x5=functools.partial(SeparableConvTranspose2d, kernel_size=5, padding=2),
            dil_conv_3x3=functools.partial(DilatedConvTranspose2d, kernel_size=3, padding=2, dilation=2),
            dil_conv_5x5=functools.partial(DilatedConvTranspose2d, kernel_size=5, padding=4, dilation=2),
            identity=functools.partial(IdentityTranspose),
            # zero=functools.partial(ZeroTranspose)
        ),
        num_nodes=6,
        num_input_nodes=2,
        num_cells=9,
        reduction_cells=[2, 5, 8],
        num_predecessors=2,
        num_channels=16,
    ).cuda()

    discriminator = DARTSDiscriminator(
        in_channels=3,
        min_resolution=4,
        num_classes=10,
        operations=dict(
            sep_conv_3x3=functools.partial(SeparableConv2d, kernel_size=3, padding=1),
            sep_conv_5x5=functools.partial(SeparableConv2d, kernel_size=5, padding=2),
            dil_conv_3x3=functools.partial(DilatedConv2d, kernel_size=3, padding=2, dilation=2),
            dil_conv_5x5=functools.partial(DilatedConv2d, kernel_size=5, padding=4, dilation=2),
            identity=functools.partial(Identity),
            # zero=functools.partial(Zero)
        ),
        num_nodes=6,
        num_input_nodes=2,
        num_cells=9,
        reduction_cells=[2, 5, 8],
        num_predecessors=2,
        num_channels=128
    ).cuda()

    criterion = CrossEntropyLoss(config.label_smoothing)

    config.global_batch_size = config.local_batch_size * distributed.get_world_size()
    config.network_optimizer.lr *= config.global_batch_size / config.global_batch_denom
    config.architecture_optimizer.lr *= config.global_batch_size / config.global_batch_denom

    generator_network_optimizer = optim.Adam(
        params=generator.network.parameters(),
        lr=config.generator_network_optimizer.lr,
        betas=config.generator_network_optimizer.betas,
        weight_decay=config.generator_network_optimizer.weight_decay
    )
    generator_architecture_optimizer = optim.Adam(
        params=generator.architecture.parameters(),
        lr=config.generator_architecture_optimizer.lr,
        betas=config.generator_architecture_optimizer.betas,
        weight_decay=config.generator_architecture_optimizer.weight_decay
    )
    discriminator_network_optimizer = optim.Adam(
        params=discriminator.network.parameters(),
        lr=config.discriminator_network_optimizer.lr,
        betas=config.discriminator_network_optimizer.betas,
        weight_decay=config.discriminator_network_optimizer.weight_decay
    )
    discriminator_architecture_optimizer = optim.Adam(
        params=discriminator.architecture.parameters(),
        lr=config.discriminator_architecture_optimizer.lr,
        betas=config.discriminator_architecture_optimizer.betas,
        weight_decay=config.discriminator_architecture_optimizer.weight_decay
    )

    trainer = DARTSGANTrainer(
        generator=generator,
        generator_networks=[generator.network],
        generator_architectures=[generator.architecture],
        discriminator=discriminator,
        discriminator_networks=[discriminator.network],
        discriminator_architectures=[discriminator.architecture],
        generator_network_optimizer=generator_network_optimizer,
        generator_architecture_optimizer=generator_architecture_optimizer,
        discriminator_network_optimizer=discriminator_network_optimizer,
        discriminator_architecture_optimizer=discriminator_architecture_optimizer,
        train_train_data_loader=train_train_data_loader,
        train_val_data_loader=train_val_data_loader,
        val_data_loader=val_data_loader,
        train_train_sampler=train_train_sampler,
        train_val_sampler=train_val_sampler,
        val_sampler=val_sampler,
        log_dir=os.path.join('log', config.name)
    )

    if config.checkpoint:
        trainer.load(config.checkpoint)

    if config.training:
        for epoch in range(trainer.epoch, config.num_epochs):
            trainer.step(epoch)
            trainer.train()
            trainer.log_architectures()
            trainer.log_histograms()
            trainer.save()

    elif config.validation:
        trainer.validate()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DARTS: Differentiable Architecture Search')
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='search/cifar10')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()

    main(args)
