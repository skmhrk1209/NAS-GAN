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

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    cuda.manual_seed(config.seed)
    cuda.set_device(distributed.get_rank() % cuda.device_count())

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
            ),
            Cutout(size=(16, 16))
        ]),
        download=True
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

    train_sampler = utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = utils.data.distributed.DistributedSampler(val_dataset)

    train_data_loader = utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.local_batch_size,
        sampler=train_sampler,
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

    model = DARTS(
        operations=dict(
            sep_conv_3x3=functools.partial(SeparableConv2d, kernel_size=3, padding=1),
            sep_conv_5x5=functools.partial(SeparableConv2d, kernel_size=5, padding=2),
            dil_conv_3x3=functools.partial(DilatedConv2d, kernel_size=3, padding=2, dilation=2),
            dil_conv_5x5=functools.partial(DilatedConv2d, kernel_size=5, padding=4, dilation=2),
            avg_pool_3x3=functools.partial(AvgPool2d, kernel_size=3, padding=1, postnormalization=False),
            max_pool_3x3=functools.partial(MaxPool2d, kernel_size=3, padding=1, postnormalization=False),
            identity=functools.partial(Identity),
            # zero=functools.partial(Zero)
        ),
        stem=[
            functools.partial(Conv2d, kernel_size=3, padding=1, stride=1, affine=True, preactivation=False),
            functools.partial(Conv2d, kernel_size=3, padding=1, stride=1, affine=True, preactivation=True)
        ],
        num_nodes=6,
        num_input_nodes=2,
        num_cells=20,
        reduction_cells=[6, 13],
        num_predecessors=2,
        num_channels=36,
        num_classes=10,
        drop_prob_fn=lambda epoch: config.drop_prob * (epoch / config.num_epochs),
        temperature_fn=lambda epoch: config.temperature ** (epoch / config.num_epochs)
    )

    checkpoint = Dict(torch.load('log/checkpoints/epoch_0'))
    model.architecture.load_state_dict(checkpoint.architecture_state_dict)
    model.build_discrete_dag()
    model.build_discrete_network()

    for parameter in model.architecture.parameters():
        parameter.requires_grad_(False)

    criterion = CrossEntropyLoss(config.label_smoothing)

    config.global_batch_size = config.local_batch_size * distributed.get_world_size()
    config.lr *= config.global_batch_size / config.global_batch_denom

    optimizer = optim.SGD(
        params=model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config.num_epochs
    )

    trainer = ClassifierTrainer(
        model=model,
        criterion=criterion,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        log_dir=os.path.join('log', config.name)
    )

    if config.checkpoint:
        trainer.load(config.checkpoint)

    if config.training:
        for epoch in range(trainer.epoch, config.num_epochs):
            trainer.train()
            trainer.validate()
            trainer.save()
            trainer.step()

    elif config.validation:
        trainer.validate()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DARTS: Differentiable Architecture Search')
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='train/cifar10')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()

    main(args)
