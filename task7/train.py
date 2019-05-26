import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model import TaskModel
from load_data import TaskDataset

import torch.nn as nn
from torch.autograd import Variable
from torch import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = {
    'num_epochs': 2,
    'batch_size': 32,
    'lr': 0.001
}

train_dataset = TaskDataset('data/blob_train_image_data/', 'data/train_sym.txt')
train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

model = TaskModel().to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=params['lr'])

total_step = len(train_dataloader)
print('total_step', total_step)

for epoch in range(params['num_epochs']):
    for i, (x, y) in enumerate(train_dataloader):
        x, y = x.to(device), y.to(device)

        # Forward
        outputs = model(x)
        loss = loss_fn(outputs, y)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, params['num_epochs'], i + 1, total_step, loss.item()))

torch.save(model.state_dict(), 'model.ckpt')
print('Finished Training')






















