import torch.nn as nn

class TaskModel(nn.Module):
    def __init__(self):
        super(TaskModel, self).__init__()
        # in_channels, out_channels, kernel_size
        # conv  [batch_size, Channel, Height, Width]
        self.conv1 = nn.Conv2d(1, 128, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(3, stride=2) # [32, 32] -> (32 - 3) / 2 + 1 [15, 15]
                                            #[15, 15] -> (15 - 3) / 2 + 1 [7, 7]
        self.prediction = nn.Linear(1152, 1) # 1152 = 128 * 3 * 3

    def forward(self, x): # [32, 32, 32] [bs, height, width]
        x = x.unsqueeze(-1).permute(0, 3, 1, 2)  # [32, 1, 32, 32] [batch_size, Channel, Height, Width]
        x = self.conv1(x) # [32, 128, 32, 32]
        x = self.pool(x) # [32, 128, 15, 15]
        x = self.conv2(x) # [32, 128, 15, 15]
        x = self.pool(x)  # [32, 128, 7, 7]
        x = self.conv3(x) # [32, 128, 7, 7]
        x = self.pool(x) # [32, 128, 3, 3]
        x = x.view(x.shape[0], -1) # [32, 1152]
        pred = self.prediction(x) # [32, 1]

        return pred.squeeze(-1) # [32]