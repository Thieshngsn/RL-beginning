import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

class subModule(nn.ModuleList):
    
    def __init__(self, am, num) -> None:
        super(subModule, self).__init__()
        self.nns = nn.ModuleList()

        for _ in range(num):
            self.nns.append(nn.Linear(am, am))
            self.nns.append(nn.Hardswish())

    def forward(self, x):
        for _, l in enumerate(self.nns):
            x = l(x)
        return x


class torchNN(nn.Module):
    def __init__(self, num_in, num_hid, num_out, num_hid_layers=1, drop_prob=0.3) -> None:
        assert num_hid_layers >= 1, 'Can\'t create a neural network with less than one hidden layer'
        super(torchNN, self).__init__()
        self.in_layer = nn.Linear(num_in, num_hid)
        self.out_layer = nn.Linear(num_hid, num_out)
        self.act = nn.Hardswish()
        self.drop = nn.Dropout(p=drop_prob)
        self.act_hid = True
        if num_hid_layers > 1:
            self.hid_layer = subModule(num_hid, num_hid_layers-1)
        else:
            self.hid_layer = nn.Linear(num_hid, num_hid)
            self.act_hid = False
    
    def forward(self, x):
        x = self.in_layer(x)
        x = self.act(x)
        x = self.hid_layer(x)
        if not self.act_hid:
            x = self.act(x)
        x = self.out_layer(x)
        x = self.act(x)
        x = self.drop(x)

        return x
    
    def backward(self, x, y):
        mse = F.mse_loss
        pred = self.forward(x)
        loss = mse(pred, y)
        print(pred)
        print(loss)
"""
batch_size = 32


train_dataset = torchvision.datasets.MNIST(root='/data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='/data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


for  i, (images, labels) in enumerate(train_loader):
    images = images.reshape(-1, 28*28)
"""

if __name__ == '__main__':
    net = torchNN(2, 5, 1, 5, 0.003)
    x = torch.FloatTensor([1,1])
    print(net(x))
    net.backward(x, torch.FloatTensor(1))
