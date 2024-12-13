import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.utils import to_undirected, add_self_loops, remove_self_loops, negative_sampling, subgraph, softmax
import copy
from torch_geometric.data import Data


class GraphEncoder(nn.Module):
    def __init__(self, gencoder_dim_list, n_layers_gencoder):
        super().__init__()
        self.gencoder_dim_list = gencoder_dim_list
        self.n_layers_gencoder = n_layers_gencoder

        self.relu = nn.ReLU()
        self.bns = nn.ModuleList()
        self.gencoders = nn.ModuleList()
        for i in range(self.n_layers_gencoder):
            self.bns.append(nn.BatchNorm1d(self.gencoder_dim_list[i]))
            self.gencoders.append(
                GCNConv(in_channels=self.gencoder_dim_list[i], out_channels=self.gencoder_dim_list[i + 1],
                          add_self_loops=False))

    def forward(self, data_input, node_embedding):
        x, edge_index, batch, edge_attr = data_input.x, data_input.edge_index, data_input.batch, data_input.edge_attr
        x_index = x[:, 0].long()
        cfipf_weight = x[:, 1]
        x = node_embedding(x_index)

        for i, gencoder in enumerate(self.gencoders):
            x = self.bns[i](x)
            x = self.relu(
                gencoder(x, edge_index, edge_attr))

        tf_idf_weight_normalized = softmax(cfipf_weight, batch)
        
        x_weighted = tf_idf_weight_normalized.unsqueeze(-1) * x
        graph_embed = global_add_pool(x_weighted, batch)

        return graph_embed