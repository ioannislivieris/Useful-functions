class LearnedSwish(nn.Module):
    def __init__(self, slope=1):
        super().__init__()
        self.slope = slope * torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.slope * x * torch.sigmoid(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)