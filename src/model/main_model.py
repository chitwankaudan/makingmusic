import torch

class MusicGeneration(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MusicGeneration, self).__init__()

        self.linear0 = torch.nn.Linear(D_in, H)
        self.linear1 = torch.nn.Linear(H, D_out)
        # self.softmax = torch.nn.Softmax(dim=1) # dimension 0 is the batch dimension

    def forward(self, x):
        x = torch.clamp(self.linear0(x), min=0)
        y_pred = self.linear1(x)

        return y_pred