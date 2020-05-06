from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from model.dcrnn_cell import DCGRUCell
import random
from base import BaseModel


class DCRNNEncoder(BaseModel):
    def __init__(self, input_dim, adj_mat, max_diffusion_step, hid_dim, num_nodes,
                 num_rnn_layers, filter_type):
        super(DCRNNEncoder, self).__init__()
        self.hid_dim = hid_dim
        self._num_rnn_layers = num_rnn_layers

        # encoding_cells = []
        encoding_cells = list()
        # the first layer has different input_dim
        encoding_cells.append(DCGRUCell(input_dim=input_dim, num_units=hid_dim, adj_mat=adj_mat,
                                        max_diffusion_step=max_diffusion_step,
                                        num_nodes=num_nodes, filter_type=filter_type))

        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(DCGRUCell(input_dim=hid_dim, num_units=hid_dim, adj_mat=adj_mat,
                                            max_diffusion_step=max_diffusion_step,
                                            num_nodes=num_nodes, filter_type=filter_type))
        self.encoding_cells = nn.ModuleList(encoding_cells)

    def forward(self, inputs, initial_hidden_state):
        # inputs shape is (seq_length, batch, num_nodes, input_dim) (12, 64, 207, 2)
        # inputs to cell is (batch, num_nodes * input_dim)
        # init_hidden_state should be (num_layers, batch_size, num_nodes*num_units) (2, 64, 207*64)
        seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))  # (12, 64, 207*2)

        current_inputs = inputs
        output_hidden = []  # the output hidden states, shape (num_layers, batch, outdim)
        for i_layer in range(self._num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]
            output_inner = []
            for t in range(seq_length):
                _, hidden_state = self.encoding_cells[i_layer](current_inputs[t, ...], hidden_state)  # (50, 207*64)
                output_inner.append(hidden_state)
            output_hidden.append(hidden_state)
            current_inputs = torch.stack(output_inner, dim=0).cuda()  # seq_len, B, ...
        # output_hidden: the hidden state of each layer at last time step, shape (num_layers, batch, outdim)
        # current_inputs: the hidden state of the top layer (seq_len, B, outdim)
        return output_hidden, current_inputs

    def init_hidden(self, batch_size):
        init_states = []  # this is a list of tuples
        for i in range(self._num_rnn_layers):
            init_states.append(self.encoding_cells[i].init_hidden(batch_size))
        # init_states shape (num_layers, batch_size, num_nodes*num_units)
        return torch.stack(init_states, dim=0)


class DCGRUDecoder(BaseModel):
    def __init__(self, input_dim, adj_mat, max_diffusion_step, num_nodes,
                 hid_dim, output_dim, num_rnn_layers, filter_type):
        super(DCGRUDecoder, self).__init__()
        self.hid_dim = hid_dim
        self._num_nodes = num_nodes  # 207
        self._output_dim = output_dim  # should be 1
        self._num_rnn_layers = num_rnn_layers

        cell = DCGRUCell(input_dim=hid_dim, num_units=hid_dim,
                         adj_mat=adj_mat, max_diffusion_step=max_diffusion_step,
                         num_nodes=num_nodes, filter_type=filter_type)
        cell_with_projection = DCGRUCell(input_dim=hid_dim, num_units=hid_dim,
                                         adj_mat=adj_mat, max_diffusion_step=max_diffusion_step,
                                         num_nodes=num_nodes, num_proj=output_dim, filter_type=filter_type)

        decoding_cells = list()
        # first layer of the decoder
        decoding_cells.append(DCGRUCell(input_dim=input_dim, num_units=hid_dim,
                                        adj_mat=adj_mat, max_diffusion_step=max_diffusion_step,
                                        num_nodes=num_nodes, filter_type=filter_type))
        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers - 1):
            decoding_cells.append(cell)
        decoding_cells.append(cell_with_projection)
        self.decoding_cells = nn.ModuleList(decoding_cells)

    def forward(self, inputs, initial_hidden_state, teacher_forcing_ratio=0.5):
        """
        :param inputs: shape should be (seq_length+1, batch_size, num_nodes, input_dim)
        :param initial_hidden_state: the last hidden state of the encoder. (num_layers, batch, outdim)
        :param teacher_forcing_ratio:
        :return: outputs. (seq_length, batch_size, num_nodes*output_dim) (12, 50, 207*1)
        """
        # inputs shape is (seq_length, batch, num_nodes, input_dim) (12, 50, 207, 1)
        # inputs to cell is (batch, num_nodes * input_dim)
        seq_length = inputs.shape[0]  # should be 13
        batch_size = inputs.shape[1]
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))  # (12+1, 50, 207*1)

        # tensor to store decoder outputs
        outputs = torch.zeros(seq_length, batch_size, self._num_nodes*self._output_dim)  # (13, 50, 207*1)
        # if rnn has only one layer
        # if self._num_rnn_layers == 1:
        #     # first input to the decoder is the GO Symbol
        #     current_inputs = inputs[0]  # (64, 207*1)
        #     hidden_state = prev_hidden_state[0]
        #     for t in range(1, seq_length):
        #         output, hidden_state = self.decoding_cells[0](current_inputs, hidden_state)
        #         outputs[t] = output  # (64, 207*1)
        #         teacher_force = random.random() < teacher_forcing_ratio
        #         current_inputs = (inputs[t] if teacher_force else output)

        current_input = inputs[0]  # the first input to the rnn is GO Symbol
        for t in range(1, seq_length):
            # hidden_state = initial_hidden_state[i_layer]  # i_layer=0, 1, ...
            next_input_hidden_state = []
            for i_layer in range(0, self._num_rnn_layers):
                hidden_state = initial_hidden_state[i_layer]
                output, hidden_state = self.decoding_cells[i_layer](current_input, hidden_state)
                current_input = output  # the input of present layer is the output of last layer
                next_input_hidden_state.append(hidden_state)  # store each layer's hidden state
            initial_hidden_state = torch.stack(next_input_hidden_state, dim=0)
            outputs[t] = output  # store the last layer's output to outputs tensor
            # perform scheduled sampling teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio  # a bool value
            current_input = (inputs[t] if teacher_force else output)

        return outputs


class DCRNNModel(BaseModel):
    def __init__(self, adj_mat, batch_size, enc_input_dim, dec_input_dim, max_diffusion_step, num_nodes,
                 num_rnn_layers, rnn_units, seq_len, output_dim, filter_type):
        super(DCRNNModel, self).__init__()
        # scaler for data normalization
        # self._scaler = scaler
        self._batch_size = batch_size

        # max_grad_norm parameter is actually defined in data_kwargs
        self._num_nodes = num_nodes  # should be 207
        self._num_rnn_layers = num_rnn_layers  # should be 2
        self._rnn_units = rnn_units  # should be 64
        self._seq_len = seq_len  # should be 12
        # use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))  # should be true
        self._output_dim = output_dim  # should be 1

        # specify a GO symbol as the start of the decoder
        self.GO_Symbol = torch.zeros(1, batch_size, num_nodes * self._output_dim, 1).cuda()

        self.encoder = DCRNNEncoder(input_dim=enc_input_dim, adj_mat=adj_mat,
                                    max_diffusion_step=max_diffusion_step,
                                    hid_dim=rnn_units, num_nodes=num_nodes,
                                    num_rnn_layers=num_rnn_layers, filter_type=filter_type)
        self.decoder = DCGRUDecoder(input_dim=dec_input_dim,
                                    adj_mat=adj_mat, max_diffusion_step=max_diffusion_step,
                                    num_nodes=num_nodes, hid_dim=rnn_units,
                                    output_dim=self._output_dim,
                                    num_rnn_layers=num_rnn_layers, filter_type=filter_type)
        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, source, target, teacher_forcing_ratio):
        # the size of source/target would be (64, 12, 207, 2)
        source = torch.transpose(source, dim0=0, dim1=1)
        target = torch.transpose(target[..., :self._output_dim], dim0=0, dim1=1)
        target = torch.cat([self.GO_Symbol, target], dim=0)

        # initialize the hidden state of the encoder
        init_hidden_state = self.encoder.init_hidden(self._batch_size).cuda()

        # last hidden state of the encoder is the context
        context, _ = self.encoder(source, init_hidden_state)  # (num_layers, batch, outdim)

        outputs = self.decoder(target, context, teacher_forcing_ratio=teacher_forcing_ratio)
        # the elements of the first time step of the outputs are all zeros.
        return outputs[1:, :, :]  # (seq_length, batch_size, num_nodes*output_dim)  (12, 64, 207*1)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def num_nodes(self):
        return self._num_nodes
