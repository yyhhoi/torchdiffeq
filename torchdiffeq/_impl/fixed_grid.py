from .solvers import FixedGridODESolver, AssoicatorFixedGridODESolver
from . import rk_common
import torch

class Euler(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        return tuple(dt * f_ for f_ in func(t, y))

    @property
    def order(self):
        return 1


class Midpoint(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        y_mid = tuple(y_ + f_ * dt / 2 for y_, f_ in zip(y, func(t, y)))
        return tuple(dt * f_ for f_ in func(t + dt / 2, y_mid))

    @property
    def order(self):
        return 2


class RK4(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        return rk_common.rk4_alt_step_func(func, t, dt, y)

    @property
    def order(self):
        return 4

class AssociatorEuler(AssoicatorFixedGridODESolver):

    def step_func(self, func, t, dt, y, w, x_all, time_index):
        """
        y0 = tuple(tensor(samples, 1, 1), )
        w = tensor(num_w, samples, 1, 1)
        x_all = tensor (num_x, t, samples, 1, 1)

        """
        x_all_t = x_all[:, time_index, :, :, :]
        dw_list = []
        w = w[0]

        for w_idx in range(w.shape[0]):
            net_input = torch.cat((x_all_t[w_idx], y[0], w[w_idx]), dim=-1)
            net_output = func(t, (net_input, ))
            dw = dt * net_output[0]
            dw_list.append((dw))
        dws = tuple(dw_list)

        dy = tuple( -y_ + torch.sum(w + x_all_t, dim=0, keepdim=False) for y_ in y)

        return dws, dy

    @property
    def order(self):
        return 1