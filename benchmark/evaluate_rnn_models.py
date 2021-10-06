import torch
import torch.nn as nn
from thop.profile import profile

input_size = 160
hidden_size = 512

models = {
    'RNNCell':
    nn.Sequential(nn.RNNCell(input_size, hidden_size)),
    'GRUCell':
    nn.Sequential(nn.GRUCell(input_size, hidden_size)),
    'LSTMCell':
    nn.Sequential(nn.LSTMCell(input_size, hidden_size)),
    'RNN':
    nn.Sequential(nn.RNN(input_size, hidden_size)),
    'GRU':
    nn.Sequential(nn.GRU(input_size, hidden_size)),
    'LSTM':
    nn.Sequential(nn.LSTM(input_size, hidden_size)),
    'stacked-RNN':
    nn.Sequential(nn.RNN(input_size, hidden_size, num_layers=4)),
    'stacked-GRU':
    nn.Sequential(nn.GRU(input_size, hidden_size, num_layers=4)),
    'stacked-LSTM':
    nn.Sequential(nn.LSTM(input_size, hidden_size, num_layers=4)),
    'BiRNN':
    nn.Sequential(nn.RNN(input_size, hidden_size, bidirectional=True)),
    'BiGRU':
    nn.Sequential(nn.GRU(input_size, hidden_size, bidirectional=True)),
    'BiLSTM':
    nn.Sequential(nn.LSTM(input_size, hidden_size, bidirectional=True)),
    'stacked-BiRNN':
    nn.Sequential(
        nn.RNN(input_size, hidden_size, bidirectional=True, num_layers=4)),
    'stacked-BiGRU':
    nn.Sequential(
        nn.GRU(input_size, hidden_size, bidirectional=True, num_layers=4)),
    'stacked-BiLSTM':
    nn.Sequential(
        nn.LSTM(input_size, hidden_size, bidirectional=True, num_layers=4)),
}

print('{} | {} | {}'.format('Model', 'Params(M)', "FLOPs(G)"))
print("---|---|---")

for name, model in models.items():
    # time_first dummy inputs
    inputs = torch.randn(100, 32, input_size)
    if name.find('Cell') != -1:
        total_ops, total_params = profile(model, (inputs[0], ), verbose=False)
    else:
        total_ops, total_params = profile(model, (inputs, ), verbose=False)
    print('{} | {:.2f} | {:.2f}'.format(
        name,
        total_params / 1e6,
        total_ops / 1e9,
    ))

# validate batch_first support
inputs = torch.randn(100, 32, input_size)
ops_time_first = profile(nn.Sequential(nn.LSTM(input_size, hidden_size)),
                         (inputs, ),
                         verbose=False)[0]
ops_batch_first = profile(nn.Sequential(
    nn.LSTM(input_size, hidden_size, batch_first=True)),
                          (inputs.transpose(0, 1), ),
                          verbose=False)[0]
assert ops_batch_first == ops_time_first