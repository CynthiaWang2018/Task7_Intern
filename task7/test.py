import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model import TaskModel
from load_data import TaskDataset, TaskDataset2

import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import numpy as np
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = {
    'batch_size': 5000,  # len(test_)
    'lr': 0.001
}

test_dataset = TaskDataset2('data/blob_test_image_data/')
test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

model = TaskModel().to(device)
model.load_state_dict(torch.load('./model.ckpt'))

total_step = len(test_dataloader)
print('total_step', total_step)

with torch.no_grad():
    for x in test_dataloader:
        x = x.to(device)
        outputs = model(x)
        outputs = F.sigmoid(outputs)
        thres = Variable(torch.Tensor([0.5])).to(device)
        predicted = (outputs > thres).float() * 1.
        predicted = predicted.cpu().detach().numpy()
        print(predicted)

df2  =pd.DataFrame({'predicted': list(predicted.reshape(-1))})

df2.to_csv('./submission.csv', header=None, index=None)
print(len(predicted))
print('Finised Testing')