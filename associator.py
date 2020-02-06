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
    func = AssociatorODEFunc()
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    n_iter = 2000
    batch_size = 32
    num_x, num_y = 1, 1

    i_list, loss_list = [], []
    gen = ExperimentSignalGenerator(num_x, num_y, batch_size=batch_size, step_size=0.01)
    testgen = ExperimentSignalGenerator(num_x, num_y, batch_size=1, step_size=0.01)

    with torch.no_grad():
        y_test = np.linspace(-1, 1, 100)
        x_test = np.linspace(-1, 1, 100)
        xx_test, yy_test = np.meshgrid(x_test, y_test)
        xx_test = xx_test.reshape(-1, 1, 1)
        yy_test = yy_test.reshape(-1, 1, 1)
        xx_test = torch.from_numpy(xx_test).float()
        yy_test = torch.from_numpy(yy_test).float()
        ww_test = torch.zeros(xx_test.shape).float()

    for i in range(n_iter):
        x, y, t = gen.generate_data(signal_input_duration=0.5, jitter=1)  # x,y shape = (num_x, t, samples, 1, 1)
        t = torch.from_numpy(t).float()
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        w0 = torch.zeros((num_x, batch_size, 1, 1)).float()
        y0 = torch.zeros((batch_size, 1, 1)).float()

        pred_y, pred_w = odeint(func, y0, w0, x, t, method=method)  # (t, samples, 1, 1)

        loss = torch.mean(torch.abs(pred_y - y))
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        print(loss_meter.avg)
        i_list.append(i)
        loss_list.append(loss_meter.avg)

        if i % 1 ==0:
            t_np = t.detach().numpy().squeeze()
            x_np = x.detach().numpy()[:, :, 0, 0, 0]
            y_np = y.detach().numpy()[:, :, 0, 0, 0]
            pred_w_np = pred_w.detach().numpy()[:, :, 0, 0, 0]
            pred_y_np = pred_y.detach().numpy()[:, 0, 0, 0]

            fig, ax = plt.subplots(3)  # (num_x, t)
            ax[0].plot(t_np, x_np[0, :], label='x')
            ax[1].plot(t_np, pred_w_np, label='w')
            ax[2].plot(t_np, y_np[0, :], label='true_y')
            ax[2].plot(t_np, pred_y_np, label='pred_y')
            for axi in range(3):
                ax[axi].legend()

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

            # Weight space
            with torch.no_grad():
                ddww = func(t, torch.cat([xx_test, yy_test, ww_test], dim=-1))
                ddww_np = ddww.detach().numpy().reshape(100, 100)
                xx_test_np, yy_test_np = xx_test.detach().numpy().reshape(100, 100), yy_test.detach().numpy().reshape(100, 100)

                fig_space, ax_space = plt.subplots()
                mappable = ax_space.pcolormesh(xx_test_np, yy_test_np, ddww_np)
                plt.colorbar(mappable)
                plt.savefig('png/color_mesh_%d.png'%(i))

                plt.close()