import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import AssociatorODEint as odeint


class AssociatorODEFunc(nn.Module):

    def __init__(self):
        super(AssociatorODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, input):
        """
        t (torch.float32):
            Time Scalar with no size
        x, y, w
            Each is tensor with shape (n_samples, *, 1)
        """
        return self.net(input)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    method = 'associator'
    end = time.time()
    func = AssociatorODEFunc()
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)


    t = torch.linspace(0, 2 * np.pi, 1000)
    x1 = torch.sin(t).reshape((t.shape[0], 1, 1, 1))
    x2 = torch.sin(2 * t).reshape((t.shape[0], 1, 1, 1))
    x = torch.stack((x1, x2))  # x'shape = (num_x, t, samples, 1, 1)
    true_y = x1.clone()
    w0 = torch.zeros((2, 1, 1, 1))
    y0 = torch.zeros((1, 1, 1))

    for i in range(200):

        pred_y, pred_w = odeint(func, y0, w0, x, t, method=method)
        loss = torch.mean(torch.abs(pred_y - true_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if i % 5 ==0:
            with torch.no_grad():
                pred_y, pred_w = odeint(func, y0, w0, x, t, method=method)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('itr %d loss = %0.5f' % (i, loss.item()))
                fig, ax = plt.subplots(3, 2)

                ax[0, 0].plot(t.detach().numpy().squeeze(), x1.detach().numpy().squeeze())
                ax[0, 0].set_title('x1')
                ax[1, 0].plot(t.detach().numpy().squeeze(), x2.detach().numpy().squeeze())
                ax[1, 0].set_title('x2')
                ax[0, 1].plot(t.detach().numpy().squeeze(), pred_w.detach().numpy().squeeze()[:, 0])
                ax[0, 1].set_title('pred_w1')
                ax[1, 1].plot(t.detach().numpy().squeeze(), pred_w.detach().numpy().squeeze()[:, 1])
                ax[1, 1].set_title('pred_w2')
                ax[2, 0].plot(t.detach().numpy().squeeze(), pred_y.detach().numpy().squeeze(), label='pred')

                ax[2, 0].plot(t.detach().numpy().squeeze(), true_y.detach().numpy().squeeze(), label='true')
                ax[2, 0].legend()
                ax[2, 1].axis('off')
                fig.suptitle('Iter %d, loss = %0.5f' % (i, loss.item()))
                plt.savefig('png/%d.png'%(i))
                plt.close()
