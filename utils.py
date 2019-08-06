import torch
from torch import nn
from torchvision import ops


class Dict(dict):
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]


def zip_longest(*iterables):
    iterators = list(map(iter, iterables))
    longest_len = max(map(len, iterables))
    for i in range(longest_len):
        items = []
        for j in range(len(iterators)):
            try:
                item = next(iterators[j])
            except StopIteration:
                iterators[j] = iter(iterables[j])
                item = next(iterators[j])
            items.append(item)
        yield tuple(items)


def apply_dict(function, dictionary):
    if isinstance(dictionary, dict):
        for key, value in dictionary.items():
            dictionary[key] = apply_dict(function, value)
        dictionary = function(dictionary)
    return dictionary


def convert_group_norm(module, num_groups):
    module_output = module
    if isinstance(module, nn.BatchNorm2d):
        module_output = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=module.num_features,
            eps=module.eps,
            affine=module.affine
        )
    for name, child in module.named_children():
        module_output.add_module(name, convert_group_norm(child, num_groups))
    del module
    return module_output


def freeze_batch_norm(module):
    module_output = module
    if isinstance(module, nn.BatchNorm2d):
        module_output = ops.misc.FrozenBatchNorm2d(module.num_features)
    for name, child in module.named_children():
        module_output.add_module(name, freeze_batch_norm(child))
    del module
    return module_output
