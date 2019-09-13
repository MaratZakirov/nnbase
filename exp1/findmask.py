#
# This experiment shows how mask emerges in ConvMask tensor
#

import torch
import torch.nn as nn
import numpy as np
import torchvision
import time

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=5, padding=1)
        self.end  = nn.Linear(in_features=1 * 18 * 18, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.end(x.reshape(-1, 1 * 18 * 18))
        x = self.sigmoid(x)
        return x

def ShowTensor(x):
    assert len(x.shape) == 2
    x = (x - x.min()) / (x.max() - x.min())
    torchvision.transforms.ToPILImage()(x).resize((800, 800)).show()

def GenRandomData(batch_size=64):
    x = torch.rand(batch_size, 1, 20, 20)
    y = torch.FloatTensor(batch_size).zero_()
    for i in range(batch_size):
        pos_prob = np.random.uniform(low=0, high=1.0)

        if pos_prob < 0.3:
            x[i, 0, 2 : 10, 4] = 1.0
            x[i, 0, 2, 4 : 12] = 1.0
            y[i] = 1.0

        #torchvision.transforms.ToPILImage()(x[i]).resize((500, 500)).show()

    return x, y

model = MyNet()
if torch.cuda.is_available():
    model = model.cuda()

loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0, lr=0.0001)
lr_scheduller = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

for epoch in range(10000):
    x, y = GenRandomData()

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()

    time.sleep(0.02)
    optimizer.zero_grad()

    out = model(x)
    Loss = loss(out, y)

    Loss.backward()

    optimizer.step()
    lr_scheduller.step(1)

    print("Epoch: ", epoch, Loss.item())

torchvision.transforms.ToPILImage()(x[0].detach().cpu()).resize((800, 800)).show()
torchvision.transforms.ToPILImage()(y[0].detach().cpu()).resize((800, 800)).show()
torchvision.transforms.ToPILImage()(out[0].detach().cpu()).resize((800, 800)).show()
