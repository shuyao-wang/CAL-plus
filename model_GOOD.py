from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, GINConv, GATConv
from gcn_conv import GCNConv
import random
import pdb


class Causal(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(
        self,
        num_features,
        num_classes,
        args,
        gfn=False,
        edge_norm=True,
    ):
        super(Causal, self).__init__()
        hidden = args.hidden
        num_conv_layers = args.layers
        self.args = args
        self.global_pool = global_add_pool
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)
        hidden_in = num_features
        self.num_classes = num_classes
        #  self.with_random = args.with_random
        hidden_out = num_classes
        # self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden,
                                 gfn=True)  # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        for i in range(num_conv_layers):
            self.convs.append(
                GINConv(
                    Sequential(Linear(hidden, hidden), BatchNorm1d(hidden),
                               ReLU(), Linear(hidden, hidden), ReLU())))

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno = BatchNorm1d(hidden)
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, xo, xc):

        xc_logis = self.context_readout_layer(xc)
        xco_logis = self.random_readout_layer(xc, xo)
        xo_logis = self.objects_readout_layer(xo)
        return xc_logis, xo_logis, xco_logis

    def forward_xo(self, data):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x.float())
        x = F.relu(self.conv_feat(x, edge_index))

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        #  edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        #  node_weight_c = node_att[:, 0]
        node_weight_o = node_att[:, 1]

        #   xc = node_weight_c.view(-1, 1) * x
        xo = node_weight_o.view(-1, 1) * x
        # xc = F.relu(self.context_convs(self.bnc(xc), edge_index,
        #                                edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index,
                                       edge_weight_o))

        # xc = self.global_pool(xc, batch)  #short cut feature(batch_size*hidden)
        xo = self.global_pool(xo, batch)  #causal feature (batch_size*hidden)

        # xc_logis = self.context_readout_layer(xc)
        # xco_logis = self.random_readout_layer(xc, xo)
        # # return xc_logis, xo_logis, xco_logis
        # xo_logis = self.objects_readout_layer(xo, train_type)
        return xo

    def eval_forward(self, data):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x.float())
        x = F.relu(self.conv_feat(x, edge_index))

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        #  edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        #  node_weight_c = node_att[:, 0]
        node_weight_o = node_att[:, 1]

        #   xc = node_weight_c.view(-1, 1) * x
        xo = node_weight_o.view(-1, 1) * x
        # xc = F.relu(self.context_convs(self.bnc(xc), edge_index,
        #                                edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index,
                                       edge_weight_o))

        #    xc = self.global_pool(xc, batch)#short cut feature(batch_size*hidden)
        xo = self.global_pool(xo, batch)  #causal feature (batch_size*hidden)

        xo_logis = self.objects_readout_layer(xo)
        return xo_logis

    def forward_xc(self, data):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x.float())
        x = F.relu(self.conv_feat(x, edge_index))

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        # edge_weight_o = edge_att[:, 1]

        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        node_weight_c = node_att[:, 0]
        # node_weight_o = node_att[:, 1]

        xc = node_weight_c.view(-1, 1) * x
        #xo = node_weight_o.view(-1, 1) * x
        xc = F.relu(self.context_convs(self.bnc(xc), edge_index,
                                       edge_weight_c))
        # xo = F.relu(self.objects_convs(self.bno(xo), edge_index,
        #edge_weight_o))

        xc = self.global_pool(xc, batch)  #short cut feature(batch_size*hidden)

        return xc

    # def _dequeue_and_enqueue(self, keys):
    #     # gather keys before updating queue
    #     batch_size = keys.shape[0]

    #     ptr = int(self.queue_ptr)
    #     # assert self.K % batch_size == 0  # for simplicity

    #     # replace the keys at ptr (dequeue and enqueue)
    #     self.queue[:, ptr:ptr + batch_size] = keys.T
    #     ptr = (ptr + batch_size) % self.K  # move pointer

    #     self.queue_ptr[0] = ptr

    def context_readout_layer(self, x):

        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):

        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)

        return x_logis

    def random_readout_layer(self, xc, xo):

        if self.args.cat_or_add == "cat":
            xo.repeat_interleave(xo.shape[0], dim=0)
            xc = xc.repeat(xc.shape[0], 1)
            x = torch.cat((xc, xo), dim=1)
        else:
            # x = xc + xo
            x = (xo.unsqueeze(1) + xc.unsqueeze(0)).contiguous().view(
                -1, self.args.hidden)
        #  x=x.view(-1, self.args.hidden)

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis
