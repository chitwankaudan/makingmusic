import torch
import torch.nn as nn

'''
    This uses 2 LSTM networks in sequence similar/identically to the paper
    https://arxiv.org/ftp/arxiv/papers/1804/1804.07300.pdf
'''
class MusicGeneration(nn.Module):
    def __init__(self, time_sequence_len, time_hidden_size):
        super(MusicGeneration, self).__init__()

        self.time_sequence_len = time_sequence_len
        self.time_hidden_size = time_hidden_size
        self.init_hidden()

        self.lstm_time0 = nn.LSTM(input_size=12 * 2, hidden_size=time_hidden_size)
        self.lstm_note0 = nn.LSTM(input_size=time_sequence_len * time_hidden_size, hidden_size=12 * 2)

        self.dropout = nn.Dropout(p=0.2)

    def init_hidden(self):
        self.hidden_time0 = (torch.ones(self.time_hidden_size, time_sequence_len), torch.ones(self.time_hidden_size, time_sequence_len))
        self.hidden_note0 = (torch.ones(self.time_hidden_size * self.time_sequence_len), torch.ones(self.time_hidden_size * self.time_sequence_len))

    def forward(self, x):
        time_outs, hidden_time_n = self.lstm_time0(x, self.hidden_time0)
        note_outs, hidden_note_n = self.lstm_note0(time_outs, self.hidden_note0)

        y_pred = torch.where(note_outs > 0.5, 1.0, 0.0)
        return y_pred