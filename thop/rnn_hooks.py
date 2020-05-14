import torch
import torch.nn as nn


def _count_rnn_cell(input_size, hidden_size, bias=True):
    # h' = \tanh(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})
    total_ops = hidden_size * (input_size + hidden_size) + hidden_size
    if bias:
        total_ops += hidden_size * 2

    return total_ops


def count_rnn_cell(m: nn.RNNCell, x: torch.Tensor, y: torch.Tensor):
    total_ops = _count_rnn_cell(m.input_size, m.hidden_size, m.bias)

    batch_size = x[0].size(0)
    total_ops *= batch_size

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def _count_gru_cell(input_size, hidden_size, bias=True):
    total_ops = 0
    # r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
    # z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
    state_ops = (hidden_size + input_size) * hidden_size + hidden_size
    if bias:
        state_ops += hidden_size * 2
    total_ops += state_ops * 2

    # n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
    total_ops += (hidden_size + input_size) * hidden_size + hidden_size
    if bias:
        total_ops += hidden_size * 2
    # r hadamard : r * (~)
    total_ops += hidden_size

    # h' = (1 - z) * n + z * h
    # hadamard hadamard add
    total_ops += hidden_size * 3

    return total_ops


def count_gru_cell(m: nn.GRUCell, x: torch.Tensor, y: torch.Tensor):
    total_ops = _count_gru_cell(m.input_size, m.hidden_size, m.bias)

    batch_size = x[0].size(0)
    total_ops *= batch_size

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def _count_lstm_cell(input_size, hidden_size, bias=True):
    total_ops = 0

    # i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
    # f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
    # o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
    # g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
    state_ops = (input_size + hidden_size) * hidden_size + hidden_size
    if bias:
        state_ops += hidden_size * 2
    total_ops += state_ops * 4

    # c' = f * c + i * g \\
    # hadamard hadamard add
    total_ops += hidden_size * 3

    # h' = o * \tanh(c') \\
    total_ops += hidden_size

    return total_ops


def count_lstm_cell(m: nn.LSTMCell, x: torch.Tensor, y: torch.Tensor):
    total_ops = _count_lstm_cell(m.input_size, m.hidden_size, m.bias)

    batch_size = x[0].size(0)
    total_ops *= batch_size

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_rnn(m: nn.RNN, x: torch.Tensor, y: torch.Tensor):
    bias = m.bias
    input_size = m.input_size
    hidden_size = m.hidden_size
    num_layers = m.num_layers

    if m.batch_first:
        batch_size = x[0].size(0)
        num_steps = x[0].size(1)
    else:
        batch_size = x[0].size(1)
        num_steps = x[0].size(0)

    total_ops = 0
    if m.bidirectional:
        total_ops += _count_rnn_cell(input_size, hidden_size, bias) * 2
    else:
        total_ops += _count_rnn_cell(input_size, hidden_size, bias)

    for i in range(num_layers - 1):
        if m.bidirectional:
            total_ops += _count_rnn_cell(hidden_size * 2, hidden_size,
                                         bias) * 2
        else:
            total_ops += _count_rnn_cell(hidden_size, hidden_size, bias)

    # time unroll
    total_ops *= num_steps
    # batch_size
    total_ops *= batch_size

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_gru(m: nn.GRU, x: torch.Tensor, y: torch.Tensor):
    bias = m.bias
    input_size = m.input_size
    hidden_size = m.hidden_size
    num_layers = m.num_layers

    if m.batch_first:
        batch_size = x[0].size(0)
        num_steps = x[0].size(1)
    else:
        batch_size = x[0].size(1)
        num_steps = x[0].size(0)

    total_ops = 0
    if m.bidirectional:
        total_ops += _count_gru_cell(input_size, hidden_size, bias) * 2
    else:
        total_ops += _count_gru_cell(input_size, hidden_size, bias)

    for i in range(num_layers - 1):
        if m.bidirectional:
            total_ops += _count_gru_cell(hidden_size * 2, hidden_size,
                                         bias) * 2
        else:
            total_ops += _count_gru_cell(hidden_size, hidden_size, bias)

    # time unroll
    total_ops *= num_steps
    # batch_size
    total_ops *= batch_size

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_lstm(m: nn.LSTM, x: torch.Tensor, y: torch.Tensor):
    bias = m.bias
    input_size = m.input_size
    hidden_size = m.hidden_size
    num_layers = m.num_layers

    if m.batch_first:
        batch_size = x[0].size(0)
        num_steps = x[0].size(1)
    else:
        batch_size = x[0].size(1)
        num_steps = x[0].size(0)

    total_ops = 0
    if m.bidirectional:
        total_ops += _count_lstm_cell(input_size, hidden_size, bias) * 2
    else:
        total_ops += _count_lstm_cell(input_size, hidden_size, bias)

    for i in range(num_layers - 1):
        if m.bidirectional:
            total_ops += _count_lstm_cell(hidden_size * 2, hidden_size,
                                          bias) * 2
        else:
            total_ops += _count_lstm_cell(hidden_size, hidden_size, bias)

    # time unroll
    total_ops *= num_steps
    # batch_size
    total_ops *= batch_size

    m.total_ops += torch.DoubleTensor([int(total_ops)])
