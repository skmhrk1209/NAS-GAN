import torch
from torch import nn
from collections import OrderedDict
from collections import abc
from itertools import islice
import operator


class BufferList(nn.Module):

    def __init__(self, buffers=None):
        super().__init__()
        if buffers is not None:
            self += buffers

    def _get_abs_index(self, idx):
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return idx

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._buffers.values())[idx])
        else:
            idx = self._get_abs_index(idx)
            return self._buffers[str(idx)]

    def __setitem__(self, idx, buffer):
        idx = self._get_abs_index(idx)
        return self.register_buffer(str(idx), buffer)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())

    def __iadd__(self, buffers):
        return self.extend(buffers)

    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, buffers):
        self.register_buffer(str(len(self)), buffers)
        return self

    def extend(self, buffers):
        for i, buffer in enumerate(buffers, len(self)):
            self.register_buffer(str(i), buffer)
        return self

    def extra_repr(self):
        lines = []
        for key, buffer in self._buffers.items():
            size_str = 'x'.join(map(str, buffer.size()))
            device_str = '' if not buffer.is_cuda else ' (GPU {})'.format(buffer.get_device())
            buffer_str = 'Buffer containing: [{} of size {}{}]'.format(torch.typename(buffer.data), size_str, device_str)
            lines.append('  (' + key + '): ' + buffer_str)
        return '\n'.join(lines)


class BufferDict(nn.Module):

    def __init__(self, buffers=None):
        super().__init__()
        if buffers is not None:
            self.update(buffers)

    def __getitem__(self, key):
        return self._buffers[key]

    def __setitem__(self, key, buffer):
        self.register_buffer(key, buffer)

    def __delitem__(self, key):
        del self._buffers[key]

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.keys())

    def __contains__(self, key):
        return key in self._buffers

    def clear(self):
        self._buffers.clear()

    def pop(self, key):
        buffer = self[key]
        del self[key]
        return buffer

    def keys(self):
        return self._buffers.keys()

    def items(self):
        return self._buffers.items()

    def values(self):
        return self._buffers.values()

    def update(self, buffers):
        if isinstance(buffers, abc.Mapping):
            if isinstance(buffers, (OrderedDict, BufferDict)):
                for key, buffer in buffers.items():
                    self[key] = buffer
            else:
                for key, buffer in sorted(buffers.items()):
                    self[key] = buffer
        else:
            for key, buffer in buffers:
                self[key] = buffer

    def extra_repr(self):
        lines = []
        for key, buffer in self._buffers.items():
            size_str = 'x'.join(map(str, buffer.size()))
            device_str = '' if not buffer.is_cuda else ' (GPU {})'.format(buffer.get_device())
            buffer_str = 'Buffer containing: [{} of size {}{}]'.format(torch.typename(buffer.data), size_str, device_str)
            lines.append('  (' + key + '): ' + buffer_str)
        return '\n'.join(lines)
