import torch
import torch.nn as nn

from model.prototypical.mem_scheme import MemoryScheme
from model.prototypical.sample_scheme import SampleScheme
from .prototypical.CoPE import Net as CoPE


class Net(torch.nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()

        self.gpu = args.cuda
        self.nt = n_tasks
        self.reg = args.memory_strength
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size

        # Dataset
        self.input_shape = args.input_shape
        self.is_cifar = args.is_cifar
        self.is_mnist = args.is_mnist

        self.C, self.H, self.W = args.CHW  # Channels/Height/Width
        self.log = False  # Activated when to log
        self.dyn_mem = args.dyn_mem  # Dynamically update size for each of the

        # Memory
        self.n_memories = args.n_memories  # number of input memories per class
        self.n_feat = n_outputs  # Output size
        self.n_classes = args.n_classes  # How many classes
        self.n_total_memories = self.n_classes * self.n_memories

        self.samples_per_task = args.samples_per_task  # In dataset splits
        self.examples_seen = 0

        # setup network
        self.net = args.net

        # setup optimizer
        if args.opt == 'sgd':
            self.opt = torch.optim.SGD(self.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                       momentum=args.momentum)
        elif args.opt == 'adam':
            self.opt = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError()

        self.ce = nn.CrossEntropyLoss()

        self.class_mem = {}  # stores prototypes (metric), and exemplars (raw) class by class
        self.task_class_map = {}  # For multi-head exp (!= task-free setting)

        self.mem_update_scheme = MemoryScheme(self, qi_mode=args.qi_mode)
        self.sample_scheme = SampleScheme(self, sample_qi_mode=args.sample_qi_mode)

    def forward(self, x, t=0):
        output = self.net(x)
        return output

    def forward_training(self, x):
        return self.forward(x)

    def get_hyperparam_list(self, args):
        return []

    def observe(self, x, t, y):
        """ Train. """
        ns_new = x.size(0)
        self.net.train()
        self.examples_seen += ns_new

        if self.gpu:
            map(torch.cuda, [x, y])

        for iter_i in range(self.n_iter):  # Number of update steps
            n_exemplars = ns_new  # Do half exemplars, half new batch
            x_init = x.clone()
            y_init = y.clone()
            update = True if iter_i == self.n_iter - 1 else False  # Only last time update

            if self.gpu:
                x_init = x_init.cuda()
                y_init = y_init.cuda()

            ymem_s = torch.Tensor()
            self.net.zero_grad()

            x_s, y_s = (torch.Tensor().cuda(), torch.LongTensor().cuda()) if self.gpu else (
                torch.Tensor(), torch.LongTensor())
            if self.n_memories > 0:
                x_s, y_s, ymem_s = self.sample_scheme(y_init, n_exemplars, self.input_shape)
                x_s, y_s, ymem_s = (x_s.cuda(), y_s.cuda(), ymem_s.cuda()) if self.gpu else (
                    x_s.cpu(), y_s.cpu(), ymem_s.cpu())

            x_ext = torch.cat([x_init, x_s])
            y_ext = torch.cat([y_init, y_s])
            if self.log:
                self.summarize_batch(y_ext, y_init, y_s)  # Summarize batch

            # Forward
            f_ext = self.forward(x_ext)  # Outputs
            self.mem_update_scheme.init_new_mem(f_ext, y_ext, self.class_mem, ns_new)  # Init new class-mems

            # Loss
            loss = self.ce(f_ext, y_ext)

            # Grad + Optimize
            loss.backward()
            self.opt.step()

            # Update prototype + mem
            if update:
                self.mem_update_scheme(x_ext, f_ext, y_ext, ymem_s, self.class_mem, ns_new)  # Update input mem

            self.checks()

    def checks(self):
        for c, cmem in self.class_mem.items():
            assert len(cmem.update_age) >= cmem.qi.shape[0]

    def summarize_batch(self, y_ext, y, y_s):
        CoPE.summarize_batch(y_ext, y, y_s)

    def summarize(self):
        print()
