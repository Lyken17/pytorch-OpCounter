import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


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


def count_rnn(m: nn.RNN, x, y):
    bias = m.bias
    input_size = m.input_size
    hidden_size = m.hidden_size
    num_layers = m.num_layers

    if isinstance(x[0], PackedSequence):
        batch_size = torch.max(x[0].batch_sizes)
        num_steps = x[0].batch_sizes.size(0)
    else:
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


def count_gru(m: nn.GRU, x, y):
    bias = m.bias
    input_size = m.input_size
    hidden_size = m.hidden_size
    num_layers = m.num_layers

    if isinstance(x[0], PackedSequence):
        batch_size = torch.max(x[0].batch_sizes)
        num_steps = x[0].batch_sizes.size(0)
    else:
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


def count_lstm(m: nn.LSTM, x, y):
    bias = m.bias
    input_size = m.input_size
    hidden_size = m.hidden_size
    num_layers = m.num_layers

    if isinstance(x[0], PackedSequence):
        batch_size = torch.max(x[0].batch_sizes)
        num_steps = x[0].batch_sizes.size(0)
    else:
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
    
def count_Transformer(m: nn.Transformer, x, y):
    total_ops = 0
    src, tgt = x
    if m.batch_first:
        num_steps = src.shape[0]
        target = tgt.shape[1]
        sequence = src.shape[1]
        embedding = src.shape[2]
    else:
        target = tgt.shape[0]
        sequence = src.shape[0]
        num_steps = src.shape[1]
        embedding = src.shape[2]
    num_head = m.nhead
    encoder_layers = m.encoder.num_layers
    decoder_layers = m.decoder.num_layers
    # dim_forward(default = 2048)
    forward = m.encoder.layers[0].linear1.out_features
    total_ops = 0

    def MultiheadAttention(bool1, num_head, num_steps, target, sequence, embedding):
        if bool1 == 0:
            # linear_q,linear_k,linear_v all N,S,E
            total_multi = 3 * sequence * embedding ** 2
            # self_attn softmax(Q*K_T/sqrt(dk))*V
            total_multi += sequence ** 4 * embedding ** 2 + \
                sequence ** 2 + sequence * (3 * sequence - 1) + 1
        elif bool1 == 1:
            # linear_q,linear_k,linear_v
            total_multi = 3 * target * embedding ** 2
            # self_attn softmax(Q*K_T/sqrt(dk))*V
            total_multi += target ** 4 * embedding ** 2 + \
                target ** 2 + target * (3 * target-1) + 1
        elif bool1 == 2:
            # linear_q,linear_k,linear_v
            total_multi = embedding ** 2 * (2 * sequence + target)
            # self_attn softmax(Q*K_T/sqrt(dk))*V
            total_multi += target * (sequence ** 2) * embedding ** 3 + \
                target * sequence + target * (3 * sequence - 1)+1
        # number of heads and batchsize
        total_multi *= total_multi * num_head*num_steps
        # print(total_multi)
        # concat
        if bool1 == 0:
            total_multi += num_steps * (sequence ** 2 * num_head * embedding)
            # print(total_multi)
        else:
            total_multi += num_steps * (target ** 2 * num_head * embedding)
        # output-> (N,S,E) or (S,N,E)
        return total_multi

    def TransformerEncoderLayer(num_head, num_steps, target, sequence, embedding):
        total_en = 0
        total_en += MultiheadAttention(0, num_head,
                                       num_steps, target, sequence, embedding)
        # linear1 in_features= embedding, outfeatures= dim_forward
        total_en += num_steps * sequence * forward * (embedding ** 2)
        # linear2
        total_en += num_steps * sequence * embedding * (forward ** 2)
        # norm1 norm2
        total_en += 2 * 2 * num_steps * embedding * sequence
        # droup out 2,3
        return total_en

    def TransformerDecoderLayer(num_head, num_steps, target, sequence, embedding):
        total_de = 0
        total_de += MultiheadAttention(1, num_head,
                                       num_steps, target, sequence, embedding)
        total_de += MultiheadAttention(2, num_head,
                                       num_steps, target, sequence, embedding)
        # linear1 linear2 fft
        total_de += num_steps * target * forward * (embedding ** 2)
        total_de += num_steps * target * embedding * (forward ** 2)
        # 3* norm
        total_de += 3 * 2 * num_steps * embedding * target
        return total_de
    total_ops = encoder_layers * TransformerEncoderLayer(num_head, num_steps, target, sequence, embedding) + \
        decoder_layers * \
        TransformerDecoderLayer(num_head, num_steps,
                                target, sequence, embedding)
    m.total_ops += torch.DoubleTensor([int(total_ops)])
