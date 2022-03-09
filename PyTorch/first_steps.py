import torch
import numpy as np

a = torch.FloatTensor(3,2)
print(a)
a = torch.FloatTensor([[1,2,3],[1,2,3]])
print(a)
n = np.zeros(shape=(3,2), dtype=np.int)
a = torch.FloatTensor(n)
print(a)