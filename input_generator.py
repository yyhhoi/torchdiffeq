import numpy as np
import matplotlib.pyplot as plt

def derivative(x, p=0, dt=0.001):
    dxdt = -x  + p
    return dxdt * dt

def time_jitter(start_time, jitter_width=0.5):
    return np.random.uniform(start_time - jitter_width, start_time + jitter_width )

class ExperimentSignalGenerator:
    def __init__(self, num_x, num_y, batch_size=256, step_size=0.001):
        self.num_x = num_x
        self.num_y = num_y
        self.step_size = step_size
        self.batch_size = batch_size


    def generate_data(self, association_time=(0, 5), recall_time=(5, 10), signal_input_duration=0.5, signal_input_amplitude=1):

        # x shape = (num_x, t, samples, 1, 1)
        # y shape = (num_y, t, samples, 1, 1)

        t = np.arange(association_time[0], recall_time[1], self.step_size)
        signal_x = np.zeros((self.num_x, t.shape[0], self.batch_size, 1, 1))
        signal_y = np.zeros((self.num_x, t.shape[0], self.batch_size, 1, 1))



        asso_signal_start_time = np.random.uniform(association_time[0], association_time[1]*0.75)

        duration_idexes = np.where((t > asso_signal_start_time) & (t < (asso_signal_start_time + signal_input_duration)))

        signal_x[duration_idexes] = signal_input_amplitude




in_max = 4
num_patterns = np.random.choice(in_max - 1) + 2  # More than 1
selected_indexes = np.random.choice(np.arange(in_max), size=(num_patterns, ), replace=False)


step_size = 0.001
start_time = 0
end_time = 5
signal_input_duration = 0.5
signal_input_amplitude = 1




t = np.arange(start_time, end_time, step_size)
start_signal_time = np.random.uniform(0, end_time*0.75)

# time_jitter(start_signal_time)

input_duration_idexes = np.where((t > start_signal_time) & (t < (start_signal_time + signal_input_duration)))

signal_input = np.zeros(t.shape)
signal_input[input_duration_idexes[0]] = signal_input_amplitude


x = 0
xout = []
for t_idx, t_each in enumerate(t):

    dx = derivative(x, p=signal_input[t_idx])

    new_x = x + dx
    xout.append(new_x)
    x = new_x

plt.plot(t, xout)
plt.show()



