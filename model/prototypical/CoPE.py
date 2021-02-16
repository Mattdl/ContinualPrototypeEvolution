import torch
from model.prototypical.mem_scheme import MemoryScheme
from model.prototypical.loss_scheme import PPPloss
from model.prototypical.p_scheme import PrototypeScheme
from model.prototypical.sample_scheme import SampleScheme


class Net(torch.nn.Module):
    """ Original implementation of Continual Prototype Evolution (CoPE)
        https://arxiv.org/pdf/2009.00919.pdf
    """

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
        self.dyn_mem = args.dyn_mem  # Dynamically update size allocated to each class

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

        self.class_mem = {}  # stores prototypes (metric), and exemplars (raw) class by class
        self.task_class_map = {}  # For multi-head exp (!= task-free setting)

        # Modular implementation
        self.lossFunc = PPPloss(self, mode=args.loss_mode, T=args.loss_T, tracker=args.tracker.loss_history)
        self.p_update_scheme = PrototypeScheme(self, p_mode=args.p_mode, p_momentum=args.p_momentum, Tp=args.Tp)
        self.mem_update_scheme = MemoryScheme(self, qi_mode=args.qi_mode)
        self.sample_scheme = SampleScheme(self, sample_qi_mode=args.sample_qi_mode)

    def forward(self, x, t):
        """ Deployment forward. Find closest prototype for each sample. """
        # nearest neighbor
        nd = self.n_feat
        ns = x.size(0)

        # Get prototypes
        seen_c = len(self.class_mem.keys())

        if seen_c == 0:
            # no exemplar in memory yet, output uniform distr. over all classes
            out = torch.Tensor(ns, self.n_classes).fill_(1.0 / self.n_classes)
            if self.gpu:
                out = out.cuda()
            return out

        means = torch.ones(seen_c, nd) * float('inf')
        if self.gpu:
            means = means.cuda()
        for c, c_mem in self.class_mem.items():
            means[c] = c_mem.prototype  # Class idx gets allocated its prototype

        # Predict to nearest
        classpred = torch.LongTensor(ns)
        preds = self._forward_eval(x).data.clone()
        for sample_idx in range(ns):  # Per class
            dist = - torch.mm(means, preds[sample_idx].view(-1, preds[sample_idx].shape[-1]).t())  # Dot product
            _, ii = dist.min(0)  # Min over batch dim
            ii = ii.squeeze()
            classpred[sample_idx] = ii.item()  # Allocate class idx

        # Convert to 1-hot
        out = torch.zeros(ns, self.n_classes)
        if self.gpu:
            out = out.cuda()
        for sample_idx in range(ns):
            out[sample_idx, classpred[sample_idx]] = 1
        return out  # return 1-of-C code, ns x nc

    def _forward_eval(self, x):
        return self.forward_training(x)

    def forward_training(self, x):
        """ Learning forward. L2-embedding normalization. """
        output = self.net(x)
        output = torch.nn.functional.normalize(output, p=2, dim=1)  # L2-embedding normalization
        return output

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
            f_ext = self.forward_training(x_ext)  # Compute Loss for all samples
            self.mem_update_scheme.init_new_mem(f_ext, y_ext, self.class_mem, ns_new)  # Init new class-mems

            # Metric space updates: prototype, metric exemplars
            self.p_update_scheme(x_ext, f_ext, y_ext, ymem_s, self.class_mem, ns_new, pre_loss=True)
            self.mem_update_scheme.update_f_mem(f_ext, y_ext, ymem_s, self.class_mem, ns_new)  # Update metric mem

            if len(self.class_mem) > 1:  # Need more than one class
                loss = self.lossFunc(f_ext, y_ext, self)

                # Grad + Optimize
                loss.backward()
                self.opt.step()

            # Update prototype + mem
            if update:
                self.p_update_scheme(x_ext, f_ext, y_ext, ymem_s, self.class_mem, ns_new)
                self.mem_update_scheme(x_ext, f_ext, y_ext, ymem_s, self.class_mem, ns_new)  # Update input mem

            self.checks()

    def checks(self):
        for c, cmem in self.class_mem.items():
            assert len(cmem.update_age) >= cmem.qi.shape[0]

    @staticmethod
    def summarize_batch(y_ext, y, y_s):
        print("BATCH SUMMARY (#class:new={},resampled={}):".format(str(y.shape[0]), str(y_s.shape[0])), end='')
        unique_labels, _ = torch.unique(y_ext).squeeze().sort()
        unique_labels = unique_labels.view(-1)
        for label_idx in range(unique_labels.size(0)):
            label = unique_labels[label_idx]

            y_idxs = (y == label).nonzero().squeeze(1)
            y_s_idxs = (y_s == label).nonzero().squeeze(1)

            new = len(y_idxs)
            resampled = len(y_s_idxs)

            print(str("{: >10}").format("(#{}:n={},r={})".format(str(label.item()), new, resampled)), end='')
        print()

    def get_all_prototypes(self):
        # All prototypes
        p_x = []
        p_y = []
        for k, k_mem in self.class_mem.items():
            p_x.append(k_mem.prototype)
            p_y.append(k)
        p_x = torch.cat(p_x).detach()
        p_y = torch.tensor(p_y).detach()
        p_x, p_y = (p_x.cuda(), p_y.cuda()) if self.gpu else (p_x, p_y)
        return p_x, p_y

    def get_hyperparam_list(self, args):
        return []

    def summarize(self):
        print()
