import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import AssociatorODEint as odeint
from input_generator import ExperimentSignalGenerator

class SigGen:
    def __init__(self):

        pass


class AssociatorODEFunc(nn.Module):

    def __init__(self):
        super(AssociatorODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 50),
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
    func1 = AssociatorODEFunc()
    func2 = AssociatorODEFunc()
    func3 = AssociatorODEFunc()
    func4 = AssociatorODEFunc()
    params = list(func1.parameters()) +  list(func2.parameters()) + list(func3.parameters()) + list(func4.parameters())
    optimizer = optim.RMSprop(params, lr=1e-3)
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    n_iter = 200
    batch_size = 32
    num_x, num_y = 4, 4

    i_list, loss_list = [], []
    gen = ExperimentSignalGenerator(num_x, num_y, batch_size=batch_size, step_size=0.01)
    testgen = ExperimentSignalGenerator(num_x, num_y, batch_size=1, step_size=0.01)
    for i in range(n_iter):
        x, y, t = gen.generate_data(jitter=2)  # x,y shape = (num_x, t, samples, 1, 1)
        t = torch.from_numpy(t).float()
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        w0 = torch.zeros((4, batch_size, 1, 1)).float()
        y0 = torch.zeros((batch_size, 1, 1)).float()

        pred_y1, pred_w1 = odeint(func1, y0, w0, x, t, method=method)  # (t, samples, 1, 1)
        pred_y2, pred_w2 = odeint(func2, y0, w0, x, t, method=method)
        pred_y3, pred_w3 = odeint(func3, y0, w0, x, t, method=method)
        pred_y4, pred_w4 = odeint(func4, y0, w0, x, t, method=method)

        pred_y = torch.stack([pred_y1, pred_y2, pred_y3, pred_y4])

        loss = torch.mean(torch.abs(pred_y - y))
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        print(loss_meter.avg)
        i_list.append(i)
        loss_list.append(loss_meter.avg)
        if i % 1 ==0:
            t_np = t.detach().numpy().squeeze()
            x_np = x.detach().numpy().squeeze()[:, :, 0]
            y_np = y.detach().numpy().squeeze()[:, :, 0]
            pred_y_np = pred_y.detach().numpy().squeeze()[:, :, 0]

            fig, ax = plt.subplots(4, 2)  # (num_x, t)
            ax[0, 0].plot(t_np, x_np[0, :], label='x1')
            ax[1, 0].plot(t_np, x_np[1, :], label='x2')
            ax[2, 0].plot(t_np, x_np[2, :], label='x3')
            ax[3, 0].plot(t_np, x_np[3, :], label='x4')
            ax[0, 1].plot(t_np, y_np[0, :], label='true_y1')
            ax[1, 1].plot(t_np, y_np[1, :], label='true_y2')
            ax[2, 1].plot(t_np, y_np[2, :], label='true_y3')
            ax[3, 1].plot(t_np, y_np[3, :], label='true_y4')
            ax[0, 1].plot(t_np, pred_y_np[0, :], label='pred_y1')
            ax[1, 1].plot(t_np, pred_y_np[1, :], label='pred_y2')
            ax[2, 1].plot(t_np, pred_y_np[2, :], label='pred_y3')
            ax[3, 1].plot(t_np, pred_y_np[3, :], label='pred_y4')
            for axi in range(4):
                for axj in range(2):
                    ax[axi, axj].legend()
            fig.suptitle('Iter %d, batch loss = %0.5f' % (i, loss_meter.avg))
            plt.savefig('png/%d.png'%(i))
            plt.close()
            try:
                fig_loss, ax_loss = plt.subplots()
                if i < 50:
                    ax_loss.plot(i_list, loss_list)
                else:
                    ax_loss.plot(i_list[:-49], loss_list[:-49])

                plt.savefig('loss_record.png')
                plt.close()
            except:
                pass