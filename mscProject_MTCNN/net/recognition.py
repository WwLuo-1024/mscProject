import numpy as np
import torch

data = torch.Tensor([[1,1],[2,2],[1,1],[3,3],[2,2],[4,4],[2,2],[1,1],[2,2],[3,3],[4,4],[1,1]])
label = torch.FloatTensor([1,2,1,3,2,4,2,1,2,3,4,1])
center = torch.Tensor([[1,1],[2,2],[3,3],[4,4]])

print(label.histc(bins=4, min=0, max=3))

# data = torch.Tensor([[1,1],[1,1],[2,2],[2,2]])
# label = torch.LongTensor([0,1,0,1])
# center = torch.Tensor([[1,1],[2,2]])
#
# print(center.index_select(dim=0, index=label))