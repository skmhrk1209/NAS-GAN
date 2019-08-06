import torch
from torch import distributed
from mpi4py import MPI
import socket
import os


def init_process_group(backend):

    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    info = dict()
    if rank == 0:
        host = socket.gethostname()
        address = socket.gethostbyname(host)
        info.update(dict(MASTER_ADDR=address, MASTER_PORT='1234'))

    info = comm.bcast(info, root=0)
    info.update(dict(WORLD_SIZE=str(world_size), RANK=str(rank)))
    os.environ.update(info)

    distributed.init_process_group(backend=backend)


def average_gradients(parameters, world_size=None):
    world_size = world_size or distributed.get_world_size()
    for parameter in parameters:
        if parameter.requires_grad:
            distributed.all_reduce(parameter.grad)
            parameter.grad /= world_size


def average_tensors(tensors, world_size=None):
    world_size = world_size or distributed.get_world_size()
    for tensor in tensors:
        distributed.all_reduce(tensor)
        tensor /= world_size
