import torch
import torch.nn as nn
import torch.nn.functional as functional

class MyModule(nn.Module):
    def __init__(self, num_in, num_out, drop_prob=0.3):
        super(MyModule, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(num_in, 5),
            nn.ReLU(),
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, num_out),
            nn.Dropout(p=drop_prob),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.pipe(x)


if __name__ == '__main__':
    net = MyModule(2,3)
    v = torch.FloatTensor([[2,3]])
    out = net(v)
    print(net)
    print(out)