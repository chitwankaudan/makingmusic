import torch
import torch.nn as nn

'''
    This version uses an LSTM provided by Pytorch in the most naive, basic
    implementation. All it does is supply the entire note matrix.
'''
class MusicGenerationV0(nn.Module):
    def __init__(self, sequence_length):
        super(MusicGeneration, self).__init__()

        self.sequence_length = sequence_length

        self.lstm0 = nn.LSTM(input_size=12 * 2, hidden_size=12 * 2, num_layers=1)
        self.dropout = nn.Dropout(p=0.2)
        # self.softmax = torch.nn.Softmax(dim=1) # dimension 0 is the batch dimension

    def forward(self, x):
        x = x.view(self.sequence_length, -1, 12 * 2)
        note_matrix_raw, internal = self.lstm0(x)

        y_pred = self.dropout(note_matrix)
        y_pred = y_pred.view(self.sequence_length, -1, 12, 2)

        return y_pred

'''
    This version uses an LSTM provided by Pytorch in the most naive, basic
    implementation. After that, it rounds the 2 floats for each note at each
    time step to either 1 or 0. Returns a float tensor.
'''
class MusicGenerationV1(nn.Module):
    def __init__(self, sequence_length):
        super(MusicGeneration, self).__init__()

        self.sequence_length = sequence_length

        self.lstm0 = nn.LSTM(input_size=12 * 2, hidden_size=12 * 2, num_layers=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(self.sequence_length, -1, 12 * 2)
        note_matrix_raw, internal = self.lstm0(x)

        note_matrix = torch.where(note_matrix > 0.5, 1.0, 0.0)

        y_pred = self.dropout(note_matrix)

        y_pred = y_pred.view(self.sequence_length, -1, 12, 2)
        return y_pred

'''
    This version uses the LSTMCell, but as I was writing the code down, I realized
    that this version is exactly the same as the next one, V3, but V3 likely runs faster
'''
class MusicGenerationV2(nn.Module):
    def __init__(self, time_sequence_len, time_hidden_size):
        super(MusicGeneration, self).__init__()

        self.time_sequence_len = time_sequence_len
        self.time_hidden_size = time_hidden_size
        self.init_hidden()

        self.lstm_cell_time0 = nn.LSTMCell(input_size=12 * 2, hidden_size=time_hidden_size)
        self.lstm_cell_note0 = nn.LSTMCell(input_size=time_sequence_len * time_hidden_size, hidden_size=12 * 2)

        self.dropout = nn.Dropout(p=0.2)

    def init_hidden(self):
        self.hidden_time0 = (torch.ones(self.time_hidden_size, time_sequence_len), torch.ones(self.time_hidden_size, time_sequence_len))
        self.hidden_note0 = (torch.ones(self.time_hidden_size * self.time_sequence_len), torch.ones(self.time_hidden_size * self.time_sequence_len))

    def time_axis(self, x):
        hx, cx = self.hidden_time0
        output = []
        for i in range(self.time_sequence_len):
            hx, cx = self.lstm_cell_time0(x[i])
            output.append(hx.unsqueeze(0))

        return torch.cat(output, dim=0)

    def note_axis(self, x):
        hx, cx = self.hidden_note0
        output = []
        for i in range(12):
            hx, cx = self.lstm_cell_note0(x[i])
            output.append(hx.unsqueeze(0))

        return torch.cat(output, dim=0)

    def forward(self, x):
        time_outs = self.time_axis(x)
        note_outs = self.note_axis(time_outs)

        y_pred = torch.where(note_outs > 0.5, 1.0, 0.0)
        return y_pred

'''
    This uses 2 LSTM networks in sequence similar/identically to the paper
    https://arxiv.org/ftp/arxiv/papers/1804/1804.07300.pdf
'''
class MusicGenerationV3(nn.Module):
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