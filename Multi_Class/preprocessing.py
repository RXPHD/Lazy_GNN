from typing import Tuple

import time
import copy

import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.data import Data

from typing import NamedTuple, List, Tuple
from torch.utils.data import DataLoader
from torch_geometric.loader import NeighborLoader

partition_fn = torch.ops.torch_sparse.partition


def metis(adj_t: SparseTensor, num_parts: int, recursive: bool = False,
          log: bool = True) -> Tuple[Tensor, Tensor]:
    r"""Computes the METIS partition of a given sparse adjacency matrix
    :obj:`adj_t`, returning its "clustered" permutation :obj:`perm` and
    corresponding cluster slices :obj:`ptr`."""

    if log:
        t = time.perf_counter()
        print(f'Computing METIS partitioning with {num_parts} parts...',
              end=' ', flush=True)

    num_nodes = adj_t.size(0)

    if num_parts <= 1:
        perm, ptr = torch.arange(num_nodes), torch.tensor([0, num_nodes])
    else:
        rowptr, col, _ = adj_t.csr()
        cluster = partition_fn(rowptr, col, None, num_parts, recursive)
        cluster, perm = cluster.sort()
        ptr = torch.ops.torch_sparse.ind2ptr(cluster, num_parts)

    if log:
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    return perm, ptr


def permute(data: Data, perm: Tensor, log: bool = True) -> Data:
    r"""Permutes a :obj:`data` object according to a given permutation
    :obj:`perm`."""

    if log:
        t = time.perf_counter()
        print('Permuting data...', end=' ', flush=True)

    data = copy.copy(data)
    for key, value in data:
        if isinstance(value, Tensor) and value.size(0) == data.num_nodes:
            data[key] = value[perm]
        elif isinstance(value, Tensor) and value.size(0) == data.num_edges:
            raise NotImplementedError
        elif isinstance(value, SparseTensor):
            data[key] = value.permute(perm)

    if log:
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    return data



class SubData(NamedTuple):
    data: Data
    batch_size: int
    n_id: Tensor  # The indices of mini-batched nodes
    offset: Tensor  # The offset of contiguous mini-batched nodes
    count: Tensor  # The number of contiguous mini-batched nodes

    def to(self, *args, **kwargs):
        return SubData(self.data.to(*args, **kwargs), self.batch_size,
                       self.n_id, self.offset, self.count)


class SubgraphLoader(DataLoader):
    r"""A simple subgraph loader that, given a pre-partioned :obj:`data` object,
    generates subgraphs from mini-batches in :obj:`ptr` (including their 1-hop
    neighbors)."""
    def __init__(self, data: Data, ptr: Tensor, batch_size: int = 1,
                 bipartite: bool = True, log: bool = True, **kwargs):

        self.data = data
        self.ptr = ptr
        self.bipartite = bipartite
        self.log = log

        n_id = torch.arange(data.num_nodes)
        batches = n_id.split((ptr[1:] - ptr[:-1]).tolist())
        batches = [(i, batches[i]) for i in range(len(batches))]

        if batch_size > 1:
            super().__init__(batches, batch_size=batch_size,
                             collate_fn=self.compute_subgraph, **kwargs)

        else:  # If `batch_size=1`, we pre-process the subgraph generation:
            if log:
                t = time.perf_counter()
                print('Pre-processing subgraphs...', end=' ', flush=True)

            data_list = list(
                DataLoader(batches, collate_fn=self.compute_subgraph,
                           batch_size=batch_size, **kwargs))

            if log:
                print(f'Done! [{time.perf_counter() - t:.2f}s]')

            super().__init__(data_list, batch_size=batch_size,
                             collate_fn=lambda x: x[0], **kwargs)


    def compute_subgraph(self, batches: List[Tuple[int, Tensor]]):
        batch_ids, n_ids = zip(*batches)
        n_id = torch.cat(n_ids, dim=0)
        batch_id = torch.tensor(batch_ids)
        mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        mask[n_id.long()] = True

        kwargs = {'batch_size': n_id.shape[0], 'num_workers': 0, 'persistent_workers': False}
        loader = NeighborLoader(self.data, input_nodes=mask,
                              num_neighbors=[-1, -1], shuffle=True, **kwargs)

        for batch in loader:
            return batch

    def __repr__(self):
        return f'{self.__class__.__name__}()'