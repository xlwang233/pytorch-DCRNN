from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from lib import utils
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionGraphConv(nn.Module):
    def __init__(self, supports, input_dim, hid_dim, num_nodes, max_diffusion_step, output_dim, bias_start=0.0):
        super(DiffusionGraphConv, self).__init__()
        num_matrices = max_diffusion_step + 1
        input_size = input_dim + hid_dim
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._supports = supports
        self.weight = nn.Parameter(torch.FloatTensor(size=(input_size*num_matrices, output_dim)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(output_dim,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self, inputs, state, output_size, bias_start=0.0):
        """
        Diffusion Graph convolution with graph matrix
        :param inputs:
        :param state:
        :param output_size:
        :param bias_start:
        :return:
        """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.shape[2]
        # dtype = inputs.dtype

        x = inputs_and_state
        x0 = torch.transpose(x, dim0=0, dim1=1)
        x0 = torch.transpose(x0, dim0=1, dim1=2)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, dim=0)

        if self._max_diffusion_step == 0:
            pass
        else:
            x1 = torch.sparse.mm(self._supports, x0)
            x = self._concat(x, x1)
            for k in range(2, self._max_diffusion_step + 1):
                x2 = 2 * torch.sparse.mm(self._supports, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        num_matrices = self._max_diffusion_step + 1  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = torch.transpose(x, dim0=0, dim1=3)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        # self.weights = torch.nn.parameter(torch.FloatTensor(size=(input_size * num_matrices, output_size)))
        # nn.init.xavier_normal_(self.weights, gain=1.414)
        x = torch.matmul(x, self.weight)  # (batch_size * self._num_nodes, output_size)
        # self.biases = nn.Parameter(torch.FloatTensor(size=(output_size,)))
        # nn.init.constant_(self.biases, val=bias_start)
        x = torch.add(x, self.biases)
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])


class DCGRUCell(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """
    def __init__(self, input_dim, num_units, adj_mat, max_diffusion_step, num_nodes,
                 num_proj=None, activation=torch.tanh, use_gc_for_ru=True):
        """
        :param num_units: the hidden dim of rnn
        :param adj_mat: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param max_diffusion_step: the max diffusion step
        :param num_nodes:
        :param num_proj: num of output dim, defaults to 1 (speed)
        :param activation: if None, don't do activation for cell state
        :param use_gc_for_ru: decide whether to use graph convolution inside rnn
        """
        super(DCGRUCell, self).__init__()
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._num_proj = num_proj
        self._use_gc_for_ru = use_gc_for_ru
        supports = utils.calculate_scaled_laplacian(adj_mat, lambda_max=None)  # scipy coo matrix
        self._supports = self._build_sparse_matrix(supports).cuda()  # to pytorch sparse tensor
        # self.register_parameter('weight', None)
        # self.register_parameter('biases', None)
        # temp_inputs = torch.FloatTensor(torch.rand((batch_size, num_nodes * input_dim)))
        # temp_state = torch.FloatTensor(torch.rand((batch_size, num_nodes * num_units)))
        # self.forward(temp_inputs, temp_state)
        self.dconv_gate = DiffusionGraphConv(supports=self._supports, input_dim=input_dim,
                                             hid_dim=num_units, num_nodes=num_nodes,
                                             max_diffusion_step=max_diffusion_step,
                                             output_dim=num_units*2)
        self.dconv_candidate = DiffusionGraphConv(supports=self._supports, input_dim=input_dim,
                                                  hid_dim=num_units, num_nodes=num_nodes,
                                                  max_diffusion_step=max_diffusion_step,
                                                  output_dim=num_units)
        if num_proj is not None:
            self.project = nn.Linear(self._num_units, self._num_proj)
    # def reset_weight_parameters(self, dim1, dim2):
    #     # self.weight = nn.Parameter(inputs.new(inputs.size()).normal_(0, 1))
    #     self.weight = nn.Parameter(torch.FloatTensor(size=(dim1, dim2)))
    #     nn.init.xavier_normal_(self.weight, gain=1.414)

    # def reset_bias_parameters(self, dim2, bias_start):
    #     self.biases = nn.Parameter(torch.FloatTensor(size=(dim2,)))
    #     nn.init.constant_(self.biases, val=bias_start)

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    def forward(self, inputs, state):
        """
        :param inputs: (B, num_nodes * input_dim)
        :param state: (B, num_nodes * num_units)
        :return:
        """
        output_size = 2 * self._num_units
        # we start with bias 1.0 to not reset and not update
        if self._use_gc_for_ru:
            fn = self.dconv_gate
        else:
            fn = self._fc
        value = torch.tanh(fn(inputs, state, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(value, split_size_or_sections=int(output_size/2), dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))
        c = self.dconv_candidate(inputs, r * state, self._num_units)  # batch_size, self._num_nodes * output_size
        if self._activation is not None:
            c = self._activation(c)
        output = new_state = u * state + (1 - u) * c
        if self._num_proj is not None:
            # apply linear projection to state
            batch_size = inputs.shape[0]
            output = torch.reshape(new_state, shape=(-1, self._num_units))  # (batch*num_nodes, num_units)
            output = torch.reshape(self.project(output), shape=(batch_size, self.output_size))  # (50, 207*1)
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    @staticmethod
    def _build_sparse_matrix(L):
        """
        build pytorch sparse tensor from scipy sparse matrix
        reference: https://stackoverflow.com/questions/50665141
        :return:
        """
        shape = L.shape
        i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
        v = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        pass

    # def _dconv(self, inputs, state, output_size, bias_start=0.0):
    #     """
    #     Diffusion Graph convolution with graph matrix
    #     :param inputs:
    #     :param state:
    #     :param output_size:
    #     :param bias_start:
    #     :return:
    #     """
    #     # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
    #     batch_size = inputs.shape[0]
    #     inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
    #     state = torch.reshape(state, (batch_size, self._num_nodes, -1))
    #     inputs_and_state = torch.cat([inputs, state], dim=2)
    #     input_size = inputs_and_state.shape[2]
    #     # dtype = inputs.dtype
    #
    #     x = inputs_and_state
    #     x0 = torch.transpose(x, dim0=0, dim1=1)
    #     x0 = torch.transpose(x0, dim0=1, dim1=2)  # (num_nodes, total_arg_size, batch_size)
    #     x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
    #     x = torch.unsqueeze(x0, dim=0)
    #
    #     if self._max_diffusion_step == 0:
    #         pass
    #     else:
    #         x1 = torch.sparse.mm(self._supports, x0)
    #         x = self._concat(x, x1)
    #         for k in range(2, self._max_diffusion_step + 1):
    #             x2 = 2 * torch.sparse.mm(self._supports, x1) - x0
    #             x = self._concat(x, x2)
    #             x1, x0 = x2, x1
    #
    #     num_matrices = self._max_diffusion_step + 1  # Adds for x itself.
    #     x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
    #     x = torch.transpose(x, dim0=0, dim1=3)  # (batch_size, num_nodes, input_size, order)
    #     x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])
    #
    #     if self.weight is None:
    #         self.reset_weight_parameters(dim1=input_size * num_matrices, dim2=output_size)
    #     if self.biases is None:
    #         self.reset_bias_parameters(dim2=output_size, bias_start=0.0)
    #     # self.weights = torch.nn.parameter(torch.FloatTensor(size=(input_size * num_matrices, output_size)))
    #     # nn.init.xavier_normal_(self.weights, gain=1.414)
    #     x = torch.matmul(x, self.weight)  # (batch_size * self._num_nodes, output_size)
    #     # self.biases = nn.Parameter(torch.FloatTensor(size=(output_size,)))
    #     # nn.init.constant_(self.biases, val=bias_start)
    #     x = torch.add(x, self.biases)
    #     # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
    #     return torch.reshape(x, [batch_size, self._num_nodes * output_size])

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        pass

    def init_hidden(self, batch_size):
        # state: (B, num_nodes * num_units)
        return torch.zeros(batch_size, self._num_nodes * self._num_units)


# class GCNSeq2Seq(nn.Module):
#     def __init__(self, g_mat, num_nodes, period_step, gcinfeat=2, gcoutfeat=16,
#                  rnn_hid=64, out_dim=1):
#         """
#         :param g: the graph for convolution
#         :param period_step: the number of past days (periodic data)
#         :param in_feat: the input dim for gcn, default to 2: (speed, time_in_day)
#         :param gcnoutdim: the output dim after gcn, default to 64: the info-fused vector
#         :param rnn_in: the input dim for seq2seq, equals to gcnoutdim
#         :param rnn_hid: the hidden dim for seq2seq, defaults to 64
#         :param rnn_out: the output dim for seq2seq, defaults to 1: (speed,)
#         """
#         self.g_mat = g_mat  # the graph matrix, which should be a scipy sparse coo matrix
#         self.g_spmat = self._build_sparse_matrix()
#         self.period_step = period_step
#         self.num_nodes = num_nodes
#         self.gcoutfeat = gcoutfeat
#         self.rnn_hid = rnn_hid
#         self.out_dim = out_dim
#         # self.gconv_weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
#
#         self.gc = GraphConvolutionLayer(g_spmat=self.g_spmat, in_feat=gcinfeat, out_feat=gcoutfeat)
#         self.Encoder = Encoder(input_dim=num_nodes*gcoutfeat, hid_dim=rnn_hid)
#         self.Decoder = Decoder(input_dim=num_nodes*gcoutfeat, hid_dim=rnn_hid, output_dim=num_nodes)
#         self.Seq2Seq = Seq2Seq(encoder=self.Encoder, decoder=self.Decoder)
#         # 不考虑period信息
#         # self.RNN_period = nn.GRU(input_size=gcnoutdim, hidden_size=rnn_hid, num_layers=1,
#         #                          bias=True)
#         self.fc = nn.Linear(in_features=rnn_hid*2, out_features=out_dim)
#
#     def forward(self, src, trg):
#         # x size: (50, 12, 207, 2)
#         # Spatial part
#         # both source and target need to be convolved
#         B, T, N, _ = src.shape
#         src = torch.transpose(src, dim0=0, dim1=1)  # to (time_step, batch, num_nodes, dim)
#         src = torch.transpose(src, dim0=1, dim1=2)  # to (time_step, num_nodes, batch, dim)
#         trg = torch.transpose(trg, dim0=0, dim1=1)  # to (time_step, batch, num_nodes, dim)
#         trg = torch.transpose(trg, dim0=1, dim1=2)  # to (time_step, num_nodes, batch, dim)
#
#         # Spatial Part
#         # go through time
#         source = []
#         for t in range(T):
#             x_t = self._gconv(src[t, :, :], output_size=self.gcnoutdim)  # should be (num_nodes, batch, output_size)
#             source.append(x_t)
#         source = torch.stack(source)  # (t, num_nodes, batch_size, output_size)
#         target = []
#         for t in range(T):
#             x_t = self._gconv(trg[t, :, :], output_size=self.gcnoutdim)  # should be (num_nodes, batch, output_size)
#             target.append(x_t)
#         target = torch.stack(target)  # (t, num_nodes, batch_size, output_size=64)
#
#         # Temporal Part
#         # reshape inputs to fit for seq2seq
#         source = torch.transpose(source, dim0=1, dim1=2)  # to (t, batch, num_nodes, output_size)
#         target = torch.transpose(target, dim0=1, dim1=2)  # same as aboove
#         source = torch.reshape(source, (T, B, -1))  # num_nodes*gcout_size is the feature dimension
#         target = torch.reshape(target, (T, B, -1))
#         outputs = self.Seq2Seq(source, target)  # (t, batch, output_dim=207)
#
#         return outputs
#
#     @staticmethod
#     def _build_sparse_matrix(self):
#         """
#         build pytorch sparse tensor from scipy sparse matrix
#         reference: https://stackoverflow.com/questions/50665141
#         :return:
#         """
#         i = torch.LongTensor(np.vstack((self.g_mat.row, self.g_mat.col)).astype(int))
#         v = torch.FloatTensor(self.g_mat.data)
#         shape = self.g_mat.shape
#         return torch.sparse.FloatTensor(i, v, torch.Size(shape))
