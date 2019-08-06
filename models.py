import torch
from torch import nn
from torch import distributed
from containers import *
from ops import *
from utils import *
from operator import *
from itertools import *
import networkx as nx
import graphviz as gv
import collections


class DARTS(nn.Module):
    """Differentiable architecture search module.
    Based on the following papers.
    1. [DARTS: Differentiable Architecture Search](https://arxiv.org/pdf/1806.09055.pdf)
    2. ...
    """

    def __init__(self, operations, num_nodes, num_input_nodes, num_cells, reduction_cells,
                 num_predecessors, num_channels, acceleration=False, drop_prob_fn=None, temperature_fn=None):
        """Build DARTS with the given operations.
        Args:
            operations (dict): Dict with name as keys and nn.Module initializer
                that takes in_channels, out_channels, and stride as arguments as values.
            num_nodes (int): Number of nodes in each cell.
            num_input_nodes (int): Number of input nodes in each cell.
            num_cells (int): Number of cells in the network.
            reduction_cells (list): List of cell index that performs spatial reduction.
            num_predecessors (int): Number of incoming edges retained in discrete architecture.
            num_channels (int): Number of channels of the first cell.
        """
        super().__init__()

        self.operations = operations
        self.num_nodes = num_nodes
        self.num_input_nodes = num_input_nodes
        self.num_cells = num_cells
        self.reduction_cells = reduction_cells
        self.num_predecessors = num_predecessors
        self.num_channels = num_channels
        self.acceleration = acceleration
        self.scheduled_drop_path = ScheduledDropPath(drop_prob_fn)
        self.scheduled_gumbel_softmax = ScheduledGumbelSoftmax(temperature_fn)

        self.build_continuous_dag()
        self.build_continuous_architecture()
        self.build_continuous_network()

    def build_continuous_dag(self):

        self.dag = Dict()

        self.dag.normal = nx.DiGraph()
        for n in range(self.num_input_nodes, self.num_nodes):
            for m in range(self.num_nodes):
                if m < n:
                    self.dag.normal.add_edge(m, n)

        self.dag.reduction = nx.DiGraph()
        for n in range(self.num_input_nodes, self.num_nodes):
            for m in range(self.num_nodes):
                if m < n:
                    self.dag.reduction.add_edge(m, n)

    def build_discrete_dag(self):

        for n in self.dag.normal.nodes():
            predecessors = sorted(zip(
                self.architecture.normal[str(n)],
                self.dag.normal.predecessors(n)
            ), key=itemgetter(0))
            for weight, m in predecessors[:-self.num_predecessors]:
                self.dag.normal.remove_edge(m, n)

        for n in self.dag.reduction.nodes():
            predecessors = sorted(zip(
                self.architecture.reduction[str(n)],
                self.dag.reduction.predecessors(n)
            ), key=itemgetter(0))
            for weight, m in predecessors[:-self.num_predecessors]:
                self.dag.reduction.remove_edge(m, n)

    def build_continuous_architecture(self):

        self.architecture = nn.ParameterDict()
        self.architecture.normal = nn.ParameterDict({**{
            str((m, n)): nn.Parameter(torch.zeros(len(self.operations)))
            for m, n in self.dag.normal.edges()
        }, **{
            str(n): nn.Parameter(torch.zeros(len(list(self.dag.normal.predecessors(n)))))
            for n in self.dag.normal.nodes()
        }})
        self.architecture.reduction = nn.ParameterDict({**{
            str((m, n)): nn.Parameter(torch.zeros(len(self.operations)))
            for m, n in self.dag.reduction.edges()
        }, **{
            str(n): nn.Parameter(torch.zeros(len(list(self.dag.reduction.predecessors(n)))))
            for n in self.dag.reduction.nodes()
        }})

        self.frequencies = BufferDict()
        self.frequencies.normal = BufferDict({**{
            str((m, n)): torch.zeros(len(self.operations))
            for m, n in self.dag.normal.edges()
        }, **{
            str(n): torch.zeros(len(list(self.dag.normal.predecessors(n))))
            for n in self.dag.normal.nodes()
        }})
        self.frequencies.reduction = BufferDict({**{
            str((m, n)): torch.zeros(len(self.operations))
            for m, n in self.dag.reduction.edges()
        }, **{
            str(n): torch.zeros(len(list(self.dag.reduction.predecessors(n))))
            for n in self.dag.reduction.nodes()
        }})

    def build_continuous_network(self):
        raise NotImplementedError

    def build_discrete_network(self):
        raise NotImplementedError

    def forward_cell(self, cell, reduction, cell_outputs, node_outputs, n):
        """forward in the given cell.
        Args:
            cell (dict): A dict with edges as keys and operations as values.
            reduction (bool): Whether the cell performs spatial reduction.
            node_outputs (dict): A dict with node as keys and its outputs as values.
                This is to avoid duplicate calculation in recursion.
            n (int): The output node in the cell.
        """
        dag = self.dag.reduction if reduction else self.dag.normal
        architecture = self.architecture.reduction if reduction else self.architecture.normal
        frequencies = self.frequencies.reduction if reduction else self.frequencies.normal

        if n not in node_outputs:
            if n in range(self.num_input_nodes):
                node_outputs[n] = cell[str((n - self.num_input_nodes, n))](cell_outputs[n - self.num_input_nodes])
            else:
                if self.continuous_mode:
                    if self.acceleration:
                        node_outputs[n] = sum(iadd(frequency, 1) and (weight / weight.detach() * sum(
                            iadd(frequency, 1) and (weight / weight.detach() * operation(self.forward_cell(cell, reduction, cell_outputs, node_outputs, m)))
                            for weight, frequency, operation in sorted(zip(self.scheduled_gumbel_softmax(architecture[str((m, n))], dim=0), frequencies[str((m, n))], cell[str((m, n))]), key=itemgetter(0))[-1:]
                        )) for weight, frequency, m in sorted(zip(self.scheduled_gumbel_softmax(architecture[str(n)], dim=0), frequencies[str(n)], dag.predecessors(n)), key=itemgetter(0))[-self.num_predecessors:])
                    else:
                        node_outputs[n] = sum(weight * sum(
                            weight * operation(self.forward_cell(cell, reduction, cell_outputs, node_outputs, m))
                            for weight, operation in zip(nn.functional.softmax(architecture[str((m, n))], dim=0), cell[str((m, n))])
                        ) for weight, m in zip(nn.functional.softmax(architecture[str(n)], dim=0), dag.predecessors(n)))
                else:
                    node_outputs[n] = sum(self.scheduled_drop_path(cell[str((m, n))](self.forward_cell(cell, reduction, cell_outputs, node_outputs, m))) for m in dag.predecessors(n))

        return node_outputs[n]

    def forward(self, input):

        output = self.network.first_conv(input)
        cell_outputs = [output] * self.num_input_nodes

        for i, cell in enumerate(self.network.cells):
            node_outputs = {}
            cell_output = torch.cat([
                self.forward_cell(cell, i in self.reduction_cells, cell_outputs, node_outputs, n)
                for n in range(self.num_input_nodes, self.num_nodes)
            ], dim=1)
            cell_outputs.append(cell_output)
            print(f'layer: {i}, shape={cell_output.shape}')

        output = self.network.last_conv(cell_output)
        print(f'layer: {i}, shape={output.shape}')

        return output

    def render(self, reduction, name, directory):

        dag = self.dag.reduction if reduction else self.dag.normal
        architecture = self.architecture.reduction if reduction else self.architecture.normal

        discrete_dag = gv.Digraph(name)
        for n in dag.nodes():
            operations = sorted(((weight, m, max((
                (weight, operation) for weight, operation
                in zip(architecture[str((m, n))], self.operations.keys()) if 'zero' not in operation
            ), key=itemgetter(0))[1]) for weight, m in zip(architecture[str(n)], dag.predecessors(n))), key=itemgetter(0))
            for weight, m, operation in operations[:-self.num_predecessors]:
                discrete_dag.edge(str(m), str(n), label='', color='white')
            for weight, m, operation in operations[-self.num_predecessors:]:
                discrete_dag.edge(str(m), str(n), label=operation, color='black')

        return discrete_dag.render(directory=directory, format='png')

    def step(self, epoch=None):
        self.scheduled_drop_path.step(epoch)
        self.scheduled_gumbel_softmax.step(epoch)


class DARTSGenerator(DARTS):

    def __init__(self, latent_size, min_resolution, out_channels, *args, **kwargs):
        self.latent_size = latent_size
        self.min_resolution = min_resolution
        self.out_channels = out_channels
        super().__init__(*args, **kwargs)

    def build_continuous_network(self):

        self.network = nn.ModuleDict()

        self.network.first_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.latent_size,
                out_channels=self.num_channels,
                kernel_size=self.min_resolution,
                bias=True
            ),
            # nn.BatchNorm2d(
            #    num_features=self.num_channels,
            #    affine=True
            # )
        )

        num_channels = self.num_channels
        out_channels = [num_channels] * self.num_input_nodes

        self.network.cells = nn.ModuleList()

        for i in range(self.num_cells):

            reduction = i in self.reduction_cells
            num_channels = num_channels >> 1 if reduction else num_channels

            dag = self.dag.reduction if reduction else self.dag.normal
            architecture = self.architecture.reduction if reduction else self.architecture.normal

            cell = nn.ModuleDict({
                # NOTE: Should be factorized reduce?
                **{
                    str((n - self.num_input_nodes, n)): ConvTranspose2d(
                        in_channels=out_channels[n - self.num_input_nodes],
                        out_channels=num_channels,
                        stride=1 << len([j for j in self.reduction_cells if k < j < i]),
                        kernel_size=1,
                        padding=0,
                        affine=False
                    ) for n, k in zip(range(self.num_input_nodes), range(i - self.num_input_nodes, i))
                },
                **{
                    str((m, n)): nn.ModuleList([
                        operation(
                            in_channels=num_channels,
                            out_channels=num_channels,
                            stride=2 if reduction and m in range(self.num_input_nodes) else 1,
                            affine=False
                        ) for operation in self.operations.values()
                    ]) for m, n in dag.edges()
                }
            })

            out_channels.append(num_channels * (self.num_nodes - self.num_input_nodes))
            self.network.cells.append(cell)

        self.network.last_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=self.out_channels,
                kernel_size=1,
                bias=True
            ),
            nn.Tanh()
        )

        self.continuous_mode = True

    def build_discrete_network(self):

        self.network = nn.ModuleDict()

        self.network.first_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.latent_size,
                out_channels=self.num_channels,
                kernel_size=self.min_resolution,
                bias=True
            ),
            # nn.BatchNorm2d(
            #    num_features=self.num_channels,
            #    affine=True
            # )
        )

        num_channels = self.num_channels
        out_channels = [num_channels] * self.num_input_nodes

        self.network.cells = nn.ModuleList()

        for i in range(self.num_cells):

            reduction = i in self.reduction_cells
            num_channels = num_channels >> 1 if reduction else num_channels

            dag = self.dag.reduction if reduction else self.dag.normal
            architecture = self.architecture.reduction if reduction else self.architecture.normal

            cell = nn.ModuleDict({
                # NOTE: Should be factorized reduce?
                **{
                    str((n - self.num_input_nodes, n)): ConvTranspose2d(
                        in_channels=out_channels[n - self.num_input_nodes],
                        out_channels=num_channels,
                        stride=1 << len([j for j in self.reduction_cells if k < j < i]),
                        kernel_size=1,
                        padding=0,
                        affine=True
                    ) for n, k in zip(range(self.num_input_nodes), range(i - self.num_input_nodes, i))
                },
                **{
                    str((m, n)): max((
                        (weight, operation) for weight, (name, operation)
                        in zip(architecture[str((m, n))], self.operations.items()) if 'zero' not in name
                    ), key=itemgetter(0))[1](
                        in_channels=num_channels,
                        out_channels=num_channels,
                        stride=2 if reduction and m in range(self.num_input_nodes) else 1,
                        affine=True
                    ) for m, n in dag.edges()
                }
            })

            out_channels.append(num_channels * (self.num_nodes - self.num_input_nodes))
            self.network.cells.append(cell)

        self.network.last_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=self.out_channels,
                kernel_size=1,
                bias=True
            ),
            nn.Tanh()
        )

        self.continuous_mode = False


class DARTSDiscriminator(DARTS):

    def __init__(self, in_channels, min_resolution, num_classes, *args, **kwargs):
        self.in_channels = in_channels
        self.min_resolution = min_resolution
        self.num_classes = num_classes
        super().__init__(*args, **kwargs)

    def build_continuous_network(self):

        self.network = nn.ModuleDict()

        self.network.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.num_channels,
                kernel_size=1,
                bias=True
            ),
            # nn.BatchNorm2d(
            #    num_features=self.num_channels,
            #    affine=True
            # )
        )

        num_channels = self.num_channels
        out_channels = [num_channels] * self.num_input_nodes

        self.network.cells = nn.ModuleList()

        for i in range(self.num_cells):

            reduction = i in self.reduction_cells
            num_channels = num_channels << 1 if reduction else num_channels

            dag = self.dag.reduction if reduction else self.dag.normal
            architecture = self.architecture.reduction if reduction else self.architecture.normal

            cell = nn.ModuleDict({
                # NOTE: Should be factorized reduce?
                **{
                    str((n - self.num_input_nodes, n)): Conv2d(
                        in_channels=out_channels[n - self.num_input_nodes],
                        out_channels=num_channels,
                        stride=1 << len([j for j in self.reduction_cells if k < j < i]),
                        kernel_size=1,
                        padding=0,
                        affine=False
                    ) for n, k in zip(range(self.num_input_nodes), range(i - self.num_input_nodes, i))
                },
                **{
                    str((m, n)): nn.ModuleList([
                        operation(
                            in_channels=num_channels,
                            out_channels=num_channels,
                            stride=2 if reduction and m in range(self.num_input_nodes) else 1,
                            affine=False
                        ) for operation in self.operations.values()
                    ]) for m, n in dag.edges()
                }
            })

            out_channels.append(num_channels * (self.num_nodes - self.num_input_nodes))
            self.network.cells.append(cell)

        self.network.last_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=self.num_classes,
                kernel_size=self.min_resolution,
                bias=True
            )
        )

        self.continuous_mode = True

    def build_discrete_network(self):

        self.network = nn.ModuleDict()

        num_channels = self.num_channels
        self.network.stem = self.stem(in_channels=self.in_channels, out_channels=num_channels)
        out_channels = [num_channels] * self.num_input_nodes

        self.network.cells = nn.ModuleList()

        for i in range(self.num_cells):

            reduction = i in self.reduction_cells
            num_channels = num_channels << 1 if reduction else num_channels

            dag = self.dag.reduction if reduction else self.dag.normal
            architecture = self.architecture.reduction if reduction else self.architecture.normal

            cell = nn.ModuleDict({
                # NOTE: Should be factorized reduce?
                **{
                    str((n - self.num_input_nodes, n)): Conv2d(
                        in_channels=out_channels[n - self.num_input_nodes],
                        out_channels=num_channels,
                        stride=1 << len([j for j in self.reduction_cells if k < j < i]),
                        kernel_size=1,
                        padding=0,
                        affine=True
                    ) for n, k in zip(range(self.num_input_nodes), range(i - self.num_input_nodes, i))
                },
                **{
                    str((m, n)): max((
                        (weight, operation) for weight, (name, operation)
                        in zip(architecture[str((m, n))], self.operations.items()) if 'zero' not in name
                    ), key=itemgetter(0))[1](
                        in_channels=num_channels,
                        out_channels=num_channels,
                        stride=2 if reduction and m in range(self.num_input_nodes) else 1,
                        affine=True
                    ) for m, n in dag.edges()
                }
            })

            out_channels.append(num_channels * (self.num_nodes - self.num_input_nodes))
            self.network.cells.append(cell)

        self.network.last_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=self.num_classes,
                kernel_size=self.min_resolution,
                bias=True
            )
        )

        self.continuous_mode = False
