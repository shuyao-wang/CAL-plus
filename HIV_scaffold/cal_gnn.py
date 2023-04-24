import torch
import torch.nn as nn
from functools import partial
from torch_geometric.nn import MessagePassing, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_mean_pool
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
# from conv_base import GNN_node_Virtualnode
from GOOD.networks.models.MolEncoders import AtomEncoder, BondEncoder
from gcn_conv import GCNConv
import pdb


class HivGINConv(MessagePassing):
    """
    Graph Isomorphism Network message passing
    Input:
        - x (Tensor): node embedding
        - edge_index (Tensor): edge connectivity information
        - edge_attr (Tensor): edge feature
    Output:
        - prediction (Tensor): output node emebedding
    """
    def __init__(self, emb_dim):
        """
        Args:
            - emb_dim (int): node embedding dimensionality
        """

        super(HivGINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim),
                                       torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp(
            (1 + self.eps) * x +
            self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GINVirtual_node(torch.nn.Module):
    def __init__(self, num_layers, emb_dim, dropout=0.1, encode_node=True):

        super(GINVirtual_node, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.encode_node = encode_node
        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layers):
            self.convs.append(HivGINConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))

    def forward(self, data):

        ### virtual node embeddings for graphs
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(
                edge_index.device))
        if self.encode_node:
            h_list = [self.atom_encoder(x)]
        else:
            h_list = [x]
        for layer in range(self.num_layers):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]
            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layers - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(
                    h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP
                virtualnode_embedding = F.dropout(
                    self.mlp_virtualnode_list[layer](
                        virtualnode_embedding_temp),
                    self.dropout,
                    training=self.training)

        node_embedding = h_list[-1]
        return node_embedding


class HivCausalGIN(torch.nn.Module):
    def __init__(self, hidden_in, hidden_out, hidden, num_layer, cls_layer=2):
        super(HivCausalGIN, self).__init__()

        self.num_classes = hidden_out
        self.global_pool = global_add_pool
        self.gnn_node = GINVirtual_node(num_layers=num_layer,
                                        emb_dim=hidden_in)

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno = BatchNorm1d(hidden)
        GConv = partial(GCNConv, edge_norm=True, gfn=False)
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        if cls_layer == 2:
            self.context_mlp = torch.nn.Sequential(BatchNorm1d(hidden),
                                                   Linear(hidden, hidden),
                                                   ReLU(), BatchNorm1d(hidden),
                                                   Linear(hidden, hidden_out))
            self.object_mlp = torch.nn.Sequential(BatchNorm1d(hidden),
                                                  Linear(hidden, hidden),
                                                  ReLU(), BatchNorm1d(hidden),
                                                  Linear(hidden, hidden_out))
            self.random_mlp = torch.nn.Sequential(BatchNorm1d(hidden),
                                                  Linear(hidden, hidden),
                                                  ReLU(), BatchNorm1d(hidden),
                                                  Linear(hidden, hidden_out))
        else:
            self.context_mlp = torch.nn.Sequential(BatchNorm1d(hidden),
                                                   Linear(hidden, hidden_out))
            self.object_mlp = torch.nn.Sequential(BatchNorm1d(hidden),
                                                  Linear(hidden, hidden_out))
            self.random_mlp = torch.nn.Sequential(BatchNorm1d(hidden),
                                                  Linear(hidden, hidden_out))

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    # def forward(self, batched_data):

    #     x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
    #     row, col = edge_index

    #     x = self.gnn_node(batched_data)

    #     edge_rep = torch.cat([x[row], x[col]], dim=-1)
    #     edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
    #     edge_weight_c = edge_att[:, 0]
    #     edge_weight_o = edge_att[:, 1]

    #     node_att = F.softmax(self.node_att_mlp(x), dim=-1)
    #     node_weight_c = node_att[:, 0]
    #     node_weight_o = node_att[:, 1]

    #     xc = node_weight_c.view(-1, 1) * x
    #     xo = node_weight_o.view(-1, 1) * x
    #     xc = F.relu(self.context_convs(self.bnc(xc), edge_index,
    #                                    edge_weight_c))
    #     xo = F.relu(self.objects_convs(self.bno(xo), edge_index,
    #                                    edge_weight_o))

    #     xc = self.global_pool(xc, batch)
    #     xo = self.global_pool(xo, batch)

    #     xc_logis = self.context_readout_layer(xc)
    #     xo_logis = self.objects_readout_layer(xo)
    #     xco_logis = self.random_readout_layer(xc, xo)
    #     return xc_logis, xo_logis, xco_logis, xo, xc

    def forward(self, xo, xc):

        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis = self.random_readout_layer(xc, xo)
        return xc_logis, xo_logis, xco_logis

    def forward_xc(self, batched_data):

        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        row, col = edge_index
        x = self.gnn_node(batched_data)
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        node_weight_c = node_att[:, 0]
        xc = node_weight_c.view(-1, 1) * x
        xc = F.relu(self.context_convs(self.bnc(xc), edge_index,
                                       edge_weight_c))
        xc = self.global_pool(xc, batch)
        return xc

    def forward_xo(self, batched_data):
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        row, col = edge_index
        x = self.gnn_node(batched_data)
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_o = edge_att[:, 1]
        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        node_weight_o = node_att[:, 1]
        xo = node_weight_o.view(-1, 1) * x
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index,
                                       edge_weight_o))
        xo = self.global_pool(xo, batch)
        return xo

    def eval_forward(self, batched_data):

        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        row, col = edge_index

        x = self.gnn_node(batched_data)

        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)

        edge_weight_o = edge_att[:, 1]

        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        node_weight_c = node_att[:, 0]
        node_weight_o = node_att[:, 1]

        xo = node_weight_o.view(-1, 1) * x

        xo = F.relu(self.objects_convs(self.bno(xo), edge_index,
                                       edge_weight_o))

        xo = self.global_pool(xo, batch)

        xo_logis = self.objects_readout_layer(xo)

        return xo_logis

    def context_readout_layer(self, x):

        x = self.context_mlp(x)
        # pdb.set_trace()
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):

        x = self.object_mlp(x)
        # x_logis = F.log_softmax(x, dim=-1)
        return x

    def random_readout_layer(self, xc, xo):

        num = xc.shape[0]
        l = [i for i in range(num)]
        random_idx = torch.tensor(l)
        x = xc[random_idx] + xo
        x = self.random_mlp(x)
        # x_logis = F.log_softmax(x, dim=-1)
        return x
