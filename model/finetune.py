# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Finetuning/iid-online/iid-offline:
    FT: is a model trained online on the stream of data without any mechanism to prevent forgetting.
    iid-online: FT 1 epoch on full iid dataset (not split into tasks)
    iid-offline: iid-online for multiple epochs
"""

import torch


class Net(torch.nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        self.net = args.net

        # setup optimizer
        self.opt = torch.optim.SGD(self.parameters(), lr=args.lr)

        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()

        self.nc_per_task = n_outputs
        self.n_outputs = n_outputs
        self.n_epochs = args.n_iter
        self.mini_batch_size = args.mini_batch_size

    def forward(self, x, t):
        output = self.net(x)
        return output

    def observe(self, x, t, y):
        self.train()
        for epoch in range(self.n_epochs):
            permutation = torch.randperm(x.size()[0])

            for i in range(0, x.size()[0], self.mini_batch_size):
                self.zero_grad()
                indices = permutation[i:i + self.mini_batch_size]
                batch_x, batch_y = x[indices], y[indices]
                ptloss = self.bce(self.forward(
                    batch_x, t),
                    batch_y)
                ptloss.backward()
                self.opt.step()

    def get_hyperparam_list(self, args):
        return []
