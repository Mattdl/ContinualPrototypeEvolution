import torch
import random


class ClassMemory(object):
    """ Operational memory for a single class."""

    def __init__(self, label, init_prototype, metric_shape, qi_len=1, gpu=True):
        if init_prototype is not None:
            assert len(init_prototype.shape) == 2, "<batch dim, feat size> required"
        assert qi_len >= 0
        assert isinstance(label, int)
        self.label = label
        self.shape = metric_shape

        self.prototype = self.init_prototype_val(self.shape) if init_prototype is None else init_prototype.cuda()
        self.p_tmp, self.p_tmp_cnt = self.init_zeros(self.shape), 0  # For update over multiple iterations
        self.q = torch.Tensor()  # Up-to-date Q used for the loss
        self.q_orig = torch.Tensor()  # Q with last updated
        self.qi_len = qi_len
        self.qi = torch.Tensor()  # Raw input imgs
        self.qi_score = torch.Tensor(self.qi_len)  # score for each input img
        self.update_age = torch.Tensor()
        self.seen_cnt = 0  # How many samples seen of this class

        if gpu:
            self.prototype = self.prototype.cuda()
            self.p_tmp = self.p_tmp.cuda()
            self.q = self.q.cuda()
            self.qi = self.qi.cuda()
            self.qi_score = self.qi_score.cuda()
            self.update_age = self.update_age.cuda()

    @staticmethod
    def init_prototype_val(feat_len):
        p = torch.nn.functional.normalize(torch.empty((1, feat_len[-1])).uniform_(0, 1), p=2, dim=1).detach()
        return p

    def init_zeros(self, feat_len=None):
        if feat_len is None:
            feat_len = self.shape
        return torch.zeros((1, feat_len[-1]))

    def __str__(self):
        res = []
        res.append("{: >5}".format("Class {}:".format(self.label)))
        res.append("{: >5} ".format(""))
        res.append("p ({}):\n {: >40} ".format(list(self.prototype.shape), str(self.prototype)))
        res.append("Mem ({}):\n {: >40} ".format(list(self.q.shape), str(self.q)))
        return "\n".join(res)


class MemoryScheme(object):
    """Memory schemes to manage operational memory.
    """
    valid_qi_modes = ['reservoir']

    def __init__(self, net, qi_mode='reservoir'):
        """
        :param qi_mode: input memory queue update mode
        """
        assert qi_mode in self.valid_qi_modes, "{} not in {}".format(qi_mode, self.valid_qi_modes)
        self.net = net
        self.qi_mode = qi_mode

        if qi_mode == 'reservoir':
            self.qi_update = self.update_reservoir_queue
            self.include_new = True  # Sample from new incoming samples in the batch
            self.include_s = False  # Can sample from replayed samples as well
        else:
            raise NotImplementedError()

    def __call__(self, x, f, labels, replay_idxs, class_mem, ns_new):
        with torch.no_grad():
            self.update_queues(x, f, labels, replay_idxs, class_mem, ns_new)

    def init_new_mem(self, f, y, class_mem, ns_new):
        """Init prototoypes of new classes."""
        f_new = f[:ns_new]
        unique_labels = torch.unique(y).squeeze().view(-1)
        for label_idx in range(unique_labels.size(0)):
            c = unique_labels[label_idx].item()
            if c not in class_mem:  # Init f_new
                class_mem[c] = ClassMemory(c, None, f_new[0].shape,
                                           qi_len=self.net.n_memories, gpu=self.net.gpu)

    def update_f_mem(self, f, y, ymem_s, class_mem, ns_new):
        """ update age of all examplars, and update representations of replayed exemplars."""
        f_s = f[ns_new:]

        # Update age count
        for c, c_mem in class_mem.items():
            c_mem.update_age += 1

        if f_s is not None and len(f_s.shape) > 0:
            y_s = y[ns_new:]
            for i in range(f_s.shape[0]):
                c = y_s[i].item()
                c_mem = class_mem[c]
                mem_idx = ymem_s[i]
                c_mem.update_age[mem_idx] = 1  # Reset age
                c_mem.q[mem_idx] = f_s[i]  # Update representation

    def update_mem_sizes(self):
        """ Update to maintain """
        assert self.net.dyn_mem
        n_seen_classes = len(self.net.class_mem)
        n_mem = int(self.net.n_total_memories / n_seen_classes)
        if n_mem == self.net.n_memories:
            return
        print('-' * 80)
        print("Updating memory capacity from {} -> {} ({} seen classes)".format(
            self.net.n_memories, n_mem, n_seen_classes))
        self.net.n_memories = n_mem
        for c, cmem in self.net.class_mem.items():
            orig_size = cmem.qi.size(0)
            perm = torch.randperm(orig_size)
            idx = perm[:n_mem]
            cmem.qi_len = n_mem
            cmem.qi = cmem.qi[idx]
            cmem.q = cmem.q[idx]

            # cmem.qi_score = cmem.qi_score[idx]
            cmem.update_age = cmem.update_age[idx]

            print("Memory of class {} cutoff from {} -> {}".format(c, orig_size, len(cmem.qi)))
        self.print_mem_stats()

    def update_queues(self, x, f, y, yr, class_mem, ns_new):
        """
        Class-specific memories are updated following defined strategy.
        """
        x_new, y_new, f_new = x[:ns_new], y[:ns_new], f[:ns_new]
        x_s, y_s, f_s = x[ns_new:], y[ns_new:], f[ns_new:]
        unique_labels = torch.unique(y).squeeze().view(-1)
        for label_idx in range(unique_labels.size(0)):
            c = unique_labels[label_idx]
            c = c.item()

            # Q input
            if self.net.n_memories > 0:
                xc, fc, fk = None, None, None
                split_idx = 0
                yrc = []

                if self.include_new:
                    idxs_newc = (y_new == c).nonzero().squeeze(dim=1)
                    xc_new = x_new[idxs_newc].detach()
                    fc_new = f_new[idxs_newc].detach()
                    idxs_newk = (y_new != c).nonzero().squeeze(dim=1)
                    fk_new = f_new[idxs_newk].detach()
                    split_idx = idxs_newc.shape[0]
                    xc = xc_new
                    fc = fc_new
                    fk = fk_new

                if self.include_s:
                    idxs_sc = (y_s == c).nonzero().squeeze(dim=1)
                    xc_s = x_s[idxs_sc].detach()
                    fc_s = f_s[idxs_sc].detach()
                    idxs_sk = (y_s != c).nonzero().squeeze(dim=1)
                    fk_s = f_s[idxs_sk].detach()
                    yrc = [] if yr is None else yr[idxs_sc]
                    xc = xc_s if xc is None else torch.cat([xc, xc_s])
                    fc = fc_s if fc is None else torch.cat([fc, fc_s])
                    fk = fk_s if fk is None else torch.cat([fk, fk_s])

                if xc is None or xc.shape[0] == 0:
                    continue

                self.qi_update(class_mem[c], xc, fc, c, split_idx, yrc, fk)

        # Cutoff based on what we have eventually
        if self.net.dyn_mem:
            self.update_mem_sizes()

    def update_reservoir_queue(self, class_mem, xc_new, fc_new, y=None, num_raw=None, replay_mem_idx=None, fk=None):
        xc_new = xc_new[:self.net.batch_size]  # Don't use replayed instances
        assert xc_new.shape[0] == fc_new.shape[0]
        class_mem.seen_cnt += xc_new.shape[0]

        if class_mem.qi.shape[0] == 0:  # Initially
            class_mem.qi = xc_new
            class_mem.q = fc_new
            class_mem.update_age = torch.ones(xc_new.shape[0])
        elif len(class_mem.qi) < class_mem.qi_len:  # Append not fully filled buffer (FIFO)
            class_mem.qi = torch.cat([class_mem.qi, xc_new])
            class_mem.q = torch.cat([class_mem.q, fc_new])
            class_mem.update_age = torch.cat([class_mem.update_age, torch.ones(xc_new.shape[0])])
        else:  # Reservoir
            x_prob = [random.randrange(class_mem.seen_cnt) for _ in range(xc_new.shape[0])]
            x_prob_idxs = [idx for idx, x in enumerate(x_prob) if x < class_mem.qi_len]
            for x_prob_idx in x_prob_idxs:
                x_mem_idx = x_prob[x_prob_idx]
                class_mem.qi[x_mem_idx] = xc_new[x_prob_idx]
                class_mem.q[x_mem_idx] = fc_new[x_prob_idx]
                class_mem.update_age[x_mem_idx] = 1
        class_mem.qi = class_mem.qi[-class_mem.qi_len:].detach()
        class_mem.q = class_mem.q[-class_mem.qi_len:].detach()
        class_mem.update_age = class_mem.update_age[-class_mem.qi_len:].detach()

    def print_mem_stats(self):
        """ Print a memory summary. """
        print("# Memory population: ", end='')
        for label, mem in self.net.class_mem.items():
            print(f"c{label}={mem.qi.shape[0]}/{mem.qi_len} ", end='')
        print()
