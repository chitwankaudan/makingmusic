import torch
import torch.nn as nn

'''
    This uses 2 LSTM networks in sequence similar/identically to the paper
    https://arxiv.org/ftp/arxiv/papers/1804/1804.07300.pdf
'''
class MusicGenerationV2(nn.Module):
    def __init__(self, time_sequence_len, batch_size, time_hidden_size, data_type=torch.DoubleTensor, device=torch.device("cpu")):
        super(MusicGenerationV2, self).__init__()

        self.num_layers = 1
        self.num_directions = 1

        self.time_sequence_len = time_sequence_len
        self.batch_size = batch_size
        self.time_hidden_size = time_hidden_size
        self.data_type = data_type
        self.device = device
        self.init_hidden()

        self.lstm_time0 = nn.ModuleList([ nn.LSTM(input_size=80, hidden_size=time_hidden_size, batch_first=True) for _ in range(78) ])
        self.lstm_note0 = nn.ModuleList([ nn.LSTM(input_size=time_hidden_size, hidden_size=2, batch_first=True) for _ in range(self.time_sequence_len) ])

        self.dropout = nn.Dropout(p=0.2)

    def init_hidden(self):
        # h_0.shape = (num_layers * num_directions, batch, hidden_size)
        # c_0.shape = (num_layers * num_directions, batch, hidden_size)
        self.hidden_time0 = (
            torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.time_hidden_size).type(self.data_type).to(self.device),
            torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.time_hidden_size).type(self.data_type).to(self.device),
        )
        self.hidden_note0 = (
            torch.zeros(self.num_layers * self.num_directions, self.batch_size, 2).type(self.data_type).to(self.device),
            torch.zeros(self.num_layers * self.num_directions, self.batch_size, 2).type(self.data_type).to(self.device),
        )

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.hidden_time0 = tuple(hx.to(*args, **kwargs) for hx in self.hidden_time0)
        self.hidden_note0 = tuple(hx.to(*args, **kwargs) for hx in self.hidden_note0)
        return self

    def forward(self, x):
        batch_size = x.shape[0]
        if batch_size < self.batch_size:
            hidden_time0 = tuple(hx[:, 0:batch_size] for hx in self.hidden_time0)
            hidden_note0 = tuple(hx[:, 0:batch_size] for hx in self.hidden_note0)
        else:
            hidden_time0 = self.hidden_time0
            hidden_note0 = self.hidden_note0

        x = x.permute(0, 2, 3, 1)

        time_outs = torch.stack([
            self.lstm_time0[i](x_i, hidden_time0)[0] for i, x_i in enumerate(torch.unbind(x, dim=3), 0)
        ], dim=3)

        x = time_outs.permute(0, 3, 2, 1)

        note_outs = torch.stack([
            self.lstm_note0[i](x_i, hidden_note0)[0] for i, x_i in enumerate(torch.unbind(x, dim=3), 0)
        ], dim=1)

        x = note_outs.view(batch_size, self.time_sequence_len, -1) # x.shape = torch.Size([10, 256, 78, 2])

        y_pred = x # torch.where(x > 0.5, torch.ones(x.shape, device=self.device), torch.zeros(x.shape, device=self.device))
        return y_pred