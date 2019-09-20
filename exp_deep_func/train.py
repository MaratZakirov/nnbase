#
# This code was made
# https://arxiv.org/pdf/1612.04799.pdf
# https://habr.com/ru/post/334380/

import numpy as np
import torch
from torch.autograd import Variable

def kernel(u, v, s, w, p):
    uv = Variable(torch.FloatTensor([u, v]))

    return s[0] + w.mv(uv).sub_(p).cos().dot(s[1:])

def integrate(fun, a, b, N=100):
    res = 0
    h = (b - a) / N

    for i in np.linspace(a, b, N):
        res += fun(a + i) * h

    return res

def V(v, n, s, w, p):
    fun = lambda u: kernel(u, v, s, w, p).mul_(u - n)
    return integrate(fun, n, n+1)

def Q(v, n, s, w, p):
    fun = lambda u: kernel(u, v, s, w, p)
    return integrate(fun, n, n+1)

def W(N, s, w, p):
    Qp = lambda v, n: Q(v, n, s, w, p)
    Vp = lambda v, n: V(v, n, s, w, p)

    W = [None] * N
    W[0] = torch.stack([Qp(v, 1) - Vp(v, 1) for v in range(1, N + 1)])
    for j in range(2, N):
        W[j-1] = torch.stack([Qp(v, j) - Vp(v, j) + Vp(v, j - 1) for v in range(1, N + 1)])
    W[N-1] = torch.stack([ Vp(v, N-1) for v in range(1, N + 1)])

    W = torch.cat(W)

    return W.view(N, N).t()

s = Variable(torch.FloatTensor([1e-5, 1, 1]), requires_grad=True)
w = Variable(torch.FloatTensor(2, 2).uniform_(-1e-5, 1e-5), requires_grad=True)
p = Variable(torch.FloatTensor(2).uniform_(-1e-5, 1e-5), requires_grad=True)

data_x_t = torch.FloatTensor(100, 3).uniform_()
data_y_t = data_x_t.mm(torch.FloatTensor([[1, 2, 3]]).t_()).view(-1)

alpha = -1e-3
for i in range(1000):
    data_x, data_y = Variable(data_x_t), Variable(data_y_t)

    Wc = W(3, s, w, p)
    y = data_x.mm(Wc).sum(1)
    loss = data_y.sub(y).pow(2).mean()

    print(loss)

    loss.backward()
    s.data.add_(s.grad.data.mul(alpha))
    s.grad.data.zero_()

    w.data.add_(w.grad.data.mul(alpha))
    w.grad.data.zero_()

    p.data.add_(p.grad.data.mul(alpha))
    p.grad.data.zero_()
