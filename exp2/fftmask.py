#
# This experiment shows how mask emerges in ConvMask tensor
#

# Using FFT to convolve
# https://stackoverflow.com/questions/43086557/convolve2d-just-by-using-numpy

import torch
import numpy as np
import torchvision
from PIL import ImageDraw, ImageFont

def ShowTensor(x, name=None):
    assert len(x.shape) == 2
    x_norm = (x - x.min()) / (x.max() - x.min())
    image = torchvision.transforms.ToPILImage()((x_norm * 255).astype(np.uint8)).resize((800, 800)).convert('RGB')
    font = ImageFont.truetype("ARIAL.TTF", 30)

    message = ('' if name == None else name)  + '\nstat:\n' + str(x.max()) + ' - ' + str(x.min()) +\
              '\nstd:  ' + str(x.std()) + '\nmean: ' + str(x.mean()) + '\ndvar: ' + str(x.std() / x.mean())

    ImageDraw.Draw(image).text((0, 0), message, (0, 255, 0), font=font)
    image.show()

def GenRandomData(batch_size=64, gsz=3):
    x = torch.rand(batch_size, 1, 20, 20) * 1.0
    y = torch.FloatTensor(batch_size, 1).zero_()
    for i in range(batch_size):
        pos_prob = np.random.uniform(low=0, high=1.0)

        if pos_prob < 10.5:
            x_sh = np.random.randint(20 - 8)
            y_sh = np.random.randint(20 - 8)
            lev = 1 + np.random.randint(4)

            if i >= 0:
                x_sh = 0
                y_sh = 0
                x[i, 0, :] = 0.0

            x[i, 0, y_sh : gsz * lev + y_sh, x_sh : x_sh + lev] = \
                torch.FloatTensor(np.random.uniform(size=(gsz * lev, lev), low=0.9, high=1.0))
            x[i, 0, y_sh : y_sh + lev, x_sh : gsz * lev + x_sh] = \
                torch.FloatTensor(np.random.uniform(size=(lev, gsz * lev), low=0.9, high=1.0))
            y[i, 0] = 1.0

        ShowTensor(x[i, 0].numpy(), "Lev: " + str(lev))
        x_i_tr = torch.FloatTensor(np.abs(np.fft.fft2(x[i].numpy())))
        ShowTensor(x_i_tr[0].numpy(), "Lev: " + str(lev))

        print("fssd")

    x_ft = np.fft.fft2(x)

    val_to_show_1 = np.abs(x_ft[0] * x_ft[55] - x_ft[0] * x_ft[12]).astype(np.float32)[0]
    val_to_show_2 = np.abs(x_ft[55] - x_ft[12]).astype(np.float32)[0]

    ShowTensor(val_to_show_1)
    ShowTensor(val_to_show_2)

    #exit()

    x_ft = x_ft[0] * x_ft[55]

    (x_ft - x_ft.transpose(1, 0, 2, 3))


    exit()

    return x, y
