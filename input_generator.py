import numpy as np
import matplotlib.pyplot as plt


def derivative(x,p=0, dt=0.001):
    dxdt = -x + p
    return dxdt * dt


def get_response_template(duration, amplitude, dt):
    xs = []
    ts = []
    ys = []
    x = 0
    y = 0
    t = 0
    x_threshold = 9e9
    y_threshold = 9e9
    while True:
        if (t < duration):
            dxs = derivative(x, p=amplitude, dt=dt)
            x = x + dxs
            xs.append(x)
            dys = derivative(y, p=x, dt=dt)
            y = y + dys
            ys.append(y)
            ts.append(t)
            x_threshold = x
        else:
            dxs = derivative(x, p=0, dt=dt)
            x = x + dxs
            xs.append(x)
            dys = derivative(y, p=x, dt=dt)
            y = y + dys
            ys.append(y)

            ts.append(t)
            if x < 0.01 * x_threshold:
                break

        t = t + dt

    template_x = np.array(xs)
    template_t = np.array(ts)
    template_y = np.array(ys)
    return template_x, template_y, template_t


def time_jitter(start_time, jitter_width=0.5, one_tail=False):
    if one_tail:
        return np.random.uniform(start_time, start_time + jitter_width * 2)
    else:
        return np.random.uniform(start_time - jitter_width, start_time + jitter_width)


def find_index_by_value(arr, val):
    """

    Args:
        arr: np.darray
            Best 1-d array.
        val: float
            Value to search for in arr.
    Returns:
        Index of arr which has value that is closest to val.
    """
    abs_diff = np.abs(arr - val)
    indexes = np.where(abs_diff == np.min(abs_diff))
    return indexes[0][0]


class ExperimentSignalGenerator:
    def __init__(self, num_x, num_y, batch_size=256, step_size=0.001):
        self.num_x = num_x
        self.num_y = num_y
        self.step_size = step_size
        self.batch_size = batch_size

    def generate_data(self, sep_time=2, signal_input_duration=0.5, signal_input_amplitude=1, jitter=0.5):
        """

        Args:
            sep_time:
            signal_input_duration:
            signal_input_amplitude:
            jitter:

        Returns:
            x (np.darray): input signal with shape (num_x, t, samples, 1, 1)
            y (np.darray): output signal with shape (num_y, t, samples, 1, 1)
            t (np.darray): times with shape (t,)
        """

        # x shape = (num_x, t, samples, 1, 1)
        # y shape = (num_y, t, samples, 1, 1)

        template_x, template_y, template_t = get_response_template(duration=signal_input_duration, amplitude=signal_input_amplitude,
                                                                    dt=self.step_size)
        template_length = template_x.shape[0]
        template_tmax = np.max(template_t)

        t_max = (2 * template_tmax) + jitter * 4 + sep_time
        t = np.arange(0, t_max, self.step_size)  # (times, )
        x = np.zeros((self.num_x, t.shape[0], self.batch_size))  # (num_x, times, batch)
        y = np.zeros((self.num_y, t.shape[0], self.batch_size))

        for sample_i in range(self.batch_size):

            # selected_indexes = self.random_pattern_indexes(self.num_x)

            asso_trial = np.random.choice([0, 1], 1)[0]


            recall_template = template_x.copy() * np.random.uniform(0.5, 1.5)

            # Association starting time, x
            x_signal_asso_starttime = time_jitter(0, jitter, one_tail=True)
            x_signal_asso_startindex = find_index_by_value(t, x_signal_asso_starttime)

            # Recall starting time, x (shared with y), also shared among different neurons
            x_signal_recall_starttime = time_jitter(time_jitter(0, jitter, one_tail=True) + template_tmax + sep_time,
                                                    jitter, one_tail=True)
            x_signal_recall_startindex = find_index_by_value(t, x_signal_recall_starttime)

            # Association starting time, y
            y_signal_asso_starttime = time_jitter(0, jitter, one_tail=True)
            y_signal_asso_startindex = find_index_by_value(t, y_signal_asso_starttime)

            # Assign recall signal to x
            x[0, x_signal_recall_startindex: (x_signal_recall_startindex + template_length),
            sample_i] = recall_template

            # Assign association signals to y by chance
            if np.random.uniform() > 0.5 and asso_trial==False:
                y[0, x_signal_asso_startindex:(x_signal_asso_startindex + template_length),
                sample_i] = template_y.copy() * np.random.uniform(0.5, 1.5)

            if asso_trial:

                # Assign association signals to x
                x[0, x_signal_asso_startindex:(x_signal_asso_startindex + template_length),
                sample_i] = template_x.copy() * np.random.uniform(0.5, 1.5)

                # Assign recall signals to y
                y[0, x_signal_recall_startindex:(x_signal_recall_startindex + template_length),
                sample_i] = template_y.copy() * np.random.uniform(0.5, 1.5)

                # Assign association signals to y
                y[0, x_signal_asso_startindex:(x_signal_asso_startindex + template_length),
                sample_i] = template_y.copy() * np.random.uniform(0.5, 1.5)



        return x.reshape((self.num_x, t.shape[0], self.batch_size, 1, 1)), y.reshape((self.num_y, t.shape[0], self.batch_size, 1, 1)), t

    def random_pattern_indexes(self, max_num):
        num_patterns = np.random.choice(max_num+ 1)   # More than 1
        selected_indexes = np.random.choice(np.arange(max_num), size=(num_patterns,), replace=False)
        return selected_indexes

if __name__ == "__main__":
    gen = ExperimentSignalGenerator(1, 1, batch_size=32)
    x, y, t = gen.generate_data(jitter=2)
    num_x = x.shape[0]
    num_y = y.shape[0]
    for i in range(x.shape[2]):
        fig, ax = plt.subplots(num_x, 2)

        for ax_i in range(ax.shape[0]):
            if ax_i < num_x:
                # ax[ax_i, 0].plot(t, x[ax_i, :, i, ].squeeze())
                ax[0].plot(t, x[ax_i, :, i, ].squeeze())
            if ax_i < num_y:
                # ax[ax_i, 1].plot(t, y[ax_i, :, i, ].squeeze())
                ax[1].plot(t, y[ax_i, :, i, ].squeeze())
        plt.show()

