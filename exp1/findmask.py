#
# This experiment shows how mask emerges in ConvMask tensor
#

import torch
import torch.nn as nn
import numpy as np
import torchvision
import time

loss = torch.nn.MSELoss()

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=5, padding=1)
        self.pool = nn.MaxPool2d(18, stride=2)
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        #torchvision.transforms.ToPILImage()(x[0, 0].detach().cpu()).resize((800, 800)).show()

        x = self.conv(x)

        #torchvision.transforms.ToPILImage()(x[0, 0].detach().cpu()).resize((800, 800)).show()
        #torchvision.transforms.ToPILImage()(self.conv.weight[0, 0].detach().cpu()).resize((800, 800)).show()

        x = self.pool(x)
        x = x.reshape(-1, 1)
        x = self.linear(x)

        return x

def ShowTensor(x):
    assert len(x.shape) == 2
    print(x.max(), x.min())
    x = (x - x.min()) / (x.max() - x.min())
    image = torchvision.transforms.ToPILImage()(x).resize((800, 800))
    image.save("temp.jpg")
    image.show()

def GenRandomData(batch_size=64):
    x = torch.rand(batch_size, 1, 20, 20)
    y = torch.FloatTensor(batch_size, 1).zero_()
    for i in range(batch_size):
        pos_prob = np.random.uniform(low=0, high=1.0)

        if pos_prob < 0.5:
            x_sh = np.random.randint(20 - 5)
            y_sh = np.random.randint(20 - 5)

            x[i, 0, y_sh: 4 + y_sh, x_sh]  = torch.FloatTensor(np.random.uniform(size=4, low=0.9, high=1.0))
            x[i, 0, y_sh, x_sh : 4 + x_sh] = torch.FloatTensor(np.random.uniform(size=4, low=0.9, high=1.0))
            y[i, 0] = 1.0

        #torchvision.transforms.ToPILImage()(x[i]).resize((500, 500)).show()

    return x, y

model = MyNet()
if torch.cuda.is_available():
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0, lr=0.01)
lr_scheduller = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

for epoch in range(1000):
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

ShowTensor(model.conv.weight[0, 0].detach().cpu())
