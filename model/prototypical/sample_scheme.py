import torch
import random
import copy


class SampleScheme(object):
    """
    Sampling scheme from replay buffer in operational memory.
    """
    valid_sample_qi_modes = ['rnd']

    def __init__(self, net, sample_qi_mode='rnd'):
        """
        :param sample_qi_mode: input memory queue update mode
        """
        assert sample_qi_mode in self.valid_sample_qi_modes, "{} not in {}".format(
            sample_qi_mode, self.valid_sample_qi_modes)
        self.net = net
        self.mode = sample_qi_mode
        self.plot = False

        if sample_qi_mode == 'rnd':
            self.qi_update = self.sample_random
        else:
            raise NotImplementedError()

        self.transform_train = None
        self.resample = False

    def __call__(self, labels, n_samples, input_shape):
        with torch.no_grad():
            return self.qi_update(labels, n_samples, input_shape)

    def sample_random(self, labels, n_samples, input_shape):
        """
        Sample random class, then random from its memory.
        No resampling.
        """
        # Determine how many mem-samples available
        q_total_cnt = 0
        free_q = {}  # idxs of which ones are free in mem queue
        classes = []
        for c, mem in self.net.class_mem.items():
            mem_cnt = mem.qi.shape[0]  # Mem cnt
            free_q[c] = list(range(0, mem_cnt))
            q_total_cnt += len(free_q[c])
            classes.append(c)

        # Randomly sample how many samples to idx per class
        free_c = copy.deepcopy(classes)
        tot_sample_cnt = 0
        sample_cnt = {c: 0 for c in classes}  # How many sampled already
        sample_max = n_samples if q_total_cnt > n_samples else q_total_cnt  # How many to sample (equally divided)
        while tot_sample_cnt < sample_max:
            c_idx = random.randrange(len(free_c))
            c = free_c[c_idx]

            if sample_cnt[c] >= len(self.net.class_mem[c].qi):  # No more memories to sample
                free_c.remove(c)
                continue
            sample_cnt[c] += 1
            tot_sample_cnt += 1

        # Actually sample
        x_s = torch.zeros((n_samples,) + input_shape)
        y_s = torch.zeros((n_samples,), dtype=torch.long)
        memidx_s = torch.zeros_like(y_s)  # To know which mem sample is chosen for the given class
        s_cnt = 0
        for c, c_cnt in sample_cnt.items():
            if c_cnt > 0:
                idxs = torch.randperm(len(self.net.class_mem[c].qi))[:c_cnt]
                s = self.net.class_mem[c].qi[idxs].clone()
                x_s[s_cnt:s_cnt + c_cnt] = s
                y_s[s_cnt:s_cnt + c_cnt].fill_(c)
                memidx_s[s_cnt:s_cnt + c_cnt] = idxs  # To know which mem samples are in here

                s_cnt += c_cnt
        assert s_cnt == tot_sample_cnt == sample_max
        cutoff = s_cnt if s_cnt < sample_max else sample_max
        x_s, y_s, memidx_s = x_s[:cutoff], y_s[:cutoff], memidx_s[:cutoff]  # Trim

        return x_s, y_s, memidx_s
