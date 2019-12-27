import torch
import torch.nn as nn

'''
x: length X batch X input_size
h_prev: num_layers X batch X hidden_size
C_prev: num_layers X batch X hidden_size
'''


def count_lstm(m: nn.LSTM, x, y):
    if m.num_layers > 1 or m.bidirectional:
        raise NotImplementedError("")

    hidden_size = m.hidden_size
    input_size = m.input_size
    # assume the layout is (length, batch, features)
    length = x[0].size(0)
    batch_size = x[0].size(1)

    muls = 0
    '''
        f_t & = \sigma(W_f \cdot z + b_f)
    '''
    muls += hidden_size * (input_size + hidden_size) * batch_size * length
    '''
        i_t & = \sigma(W_i \cdot z + b_i)
    '''
    muls += hidden_size * (input_size + hidden_size) * batch_size * length
    '''
        g_t & = tanh(W_C \cdot z + b_C)
    '''
    muls += hidden_size * (input_size + hidden_size) * batch_size * length
    '''
        o_t & = \sigma(W_o \cdot z + b_t)
    '''
    muls += hidden_size * (input_size + hidden_size) * batch_size * length

    '''
        C_t & = f_t * C_{t-1} + i_t * g_t
    '''
    muls += 2 * hidden_size * batch_size
    '''
        h_t &= o_t * tanh(C_t)
    '''
    muls += hidden_size * 2

    m.total_ops += muls


if __name__ == '__main__':
    net = nn.LSTM(input_size=10, hidden_size=10)
    print(net)
