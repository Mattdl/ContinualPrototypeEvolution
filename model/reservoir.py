"""This is reservoir sampling, each sample has storage-probability 'buffer samples M / seen samples'
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random


class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        self.net = args.net

        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs

        self.opt = optim.SGD(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.n_sampled_memories = args.n_sampled_memories
        self.n_constraints = args.n_constraints
        self.gpu = args.cuda

        self.batch_size = args.batch_size
        self.n_iter = args.n_iter

        # allocate ring buffer
        self.x_buffer = []
        self.y_buffer = []

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.seen_cnt = 0

    def forward(self, x, t=0):
        output = self.net(x)
        return output

    def observe(self, x, t, y):
        """ Train. """
        # Update ring buffer storing examples from current task
        bsz = y.data.size(0)

        # Step over new batch and batch from memory
        for iter_i in range(self.n_iter):
            self.zero_grad()
            x_init = x.clone()
            y_init = y.clone()
            if self.gpu:
                x_init = x_init.cuda()
                y_init = y_init.cuda()

            sample_bs = bsz
            if self.n_memories > 0 and len(self.x_buffer) > 0:  # Sample
                perm = torch.randperm(len(self.x_buffer))
                idx = perm[:sample_bs]
                x_s = torch.stack(self.x_buffer)[idx]
                y_s = torch.stack(self.y_buffer)[idx]
                x_s, y_s = (x_s.cuda(), y_s.cuda()) if self.gpu else (x_s.cpu(), y_s.cpu())
                x_ext = torch.cat([x_init, x_s])
                y_ext = torch.cat([y_init, y_s])
            else:
                x_ext = x_init
                y_ext = y_init

            loss = self.ce(self.forward(x_ext), y_ext)
            loss.backward()
            self.opt.step()

        # Update buffers
        for i in range(bsz):
            if self.seen_cnt < self.n_memories:
                self.x_buffer.append(x[i])
                self.y_buffer.append(y[i])
            else:
                j = random.randrange(self.seen_cnt)
                if j < self.n_memories:
                    self.x_buffer[j] = x[i]
                    self.y_buffer[j] = y[i]
            self.seen_cnt += 1

        assert len(self.x_buffer) <= self.n_memories
        assert len(self.x_buffer) == len(self.y_buffer)

    def get_hyperparam_list(self, args):
        return []
