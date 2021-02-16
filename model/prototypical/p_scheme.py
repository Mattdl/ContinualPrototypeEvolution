import torch


class PrototypeScheme(object):
    valid_p_modes = ['batch_only', 'batch_momentum', 'batch_momentum_it', 'batch_momentum_incr']

    def __init__(self, net, p_mode, p_momentum=0, Tp=0.1):
        """
        :param p_mode: prototypes update mode
        """
        assert p_mode in self.valid_p_modes, "{} not in {}".format(p_mode, self.valid_p_modes)
        self.net = net
        self.Tp = Tp
        self.p_momentum = p_momentum
        self.update_pre_loss = False
        self.update_post_loss = True

        if p_mode == 'batch_only':
            self.p_update = self.update_batch_momentum
            self.p_momentum = 0  # Old value is zero, only current batch counts
        elif p_mode == 'batch_momentum':
            self.p_update = self.update_batch_momentum
        elif p_mode == 'batch_momentum_it':
            self.p_update = self.update_batch_momentum
            self.update_pre_loss = True  # Update each iteration
            self.update_post_loss = False
        elif p_mode == 'batch_momentum_incr':
            self.p_update = self.update_batch_momentum_incr
            self.update_pre_loss = True  # Accumulate batch info
            self.update_post_loss = True  # -> Only actual update
        else:
            raise NotImplementedError()

    def __call__(self, x, f, y, yr, class_mem, ns_new, pre_loss=False):
        with torch.no_grad():
            if (pre_loss and self.update_pre_loss) or \
                    (not pre_loss and self.update_post_loss):
                if self.net.gpu:
                    map(torch.cuda, [x, f, y, yr])
                self.p_update(x, f, y, yr, class_mem, ns_new, pre_loss)

    def summarize_p_update(self, c, new_p, old_p):
        d = (new_p - old_p).pow_(2).sum(1).sqrt_()
        if self.net.log:
            print("Class {} p-update: L2 delta={:.4f}".format(c, float(d.mean().item())))

    @staticmethod
    def momentum_update(old_value, new_value, momentum, debug=False):
        update = momentum * old_value + (1 - momentum) * new_value
        if debug:
            print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
                momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
                torch.norm(update, p=2)), end='')
        return update

    def update_batch_momentum_incr(self, x, f, y, yr, class_mem, ns_new, pre_loss):
        x_new, y_new, f_new = x[:ns_new], y[:ns_new], f[:ns_new]
        x_s, y_s, f_s = x[ns_new:], y[ns_new:], f[ns_new:]

        if pre_loss and len(y_s) > 0:  # Accumulate replayed exemplars (/class)
            unique_labels = torch.unique(y_s)
            for label_idx in range(unique_labels.size(0)):
                c = unique_labels[label_idx]
                idxs = (y_s == c).nonzero().squeeze(1)

                p_tmp_batch = f_s[idxs].sum(dim=0).unsqueeze(0)
                if self.net.gpu:
                    p_tmp_batch = p_tmp_batch.cuda()
                class_mem[c.item()].p_tmp += p_tmp_batch
                class_mem[c.item()].p_tmp_cnt += len(idxs)
        else:
            for c, cmem in class_mem.items():
                # Include new ones too (All replayed already in pre-loss)
                idxs_new = (y_new == c).nonzero().squeeze(1)
                if len(idxs_new) > 0:
                    p_tmp_batch = f_new[idxs_new].sum(dim=0).unsqueeze(0)
                    if self.net.gpu:
                        p_tmp_batch = p_tmp_batch.cuda()
                    class_mem[c].p_tmp += p_tmp_batch
                    class_mem[c].p_tmp_cnt += len(idxs_new)

                if class_mem[c].p_tmp_cnt > 0:
                    incr_p = class_mem[c].p_tmp / class_mem[c].p_tmp_cnt
                    old_p = class_mem[c].prototype.clone()
                    new_p_momentum = self.momentum_update(old_p, incr_p, self.p_momentum)
                    new_p = torch.nn.functional.normalize(new_p_momentum, p=2,
                                                          dim=1).detach()  # L2-embedding normalization
                    self.summarize_p_update(c, new_p, old_p)

                    # Update
                    class_mem[c].prototype = new_p
                    assert not torch.isnan(class_mem[c].prototype).any()

                    # Re-init
                    class_mem[c].p_tmp = class_mem[c].init_zeros().cuda() if self.net.gpu else class_mem[c].init_zeros()
                    class_mem[c].p_tmp_cnt = 0

    def update_batch_momentum(self, x, f, y, yr, class_mem, ns_new, pre_loss):
        """Take momentum of current batch avg with prev prototype (based on prev batches)."""
        if self.net.log:
            print()
            print("* Updating Prototype *")
        unique_labels = torch.unique(y).squeeze()
        for label_idx in range(unique_labels.size(0)):
            c = unique_labels[label_idx]
            idxs = (y == c).nonzero().squeeze(1)

            batch_p = f[idxs].mean(0).unsqueeze(0)  # Mean of whole batch
            old_p = class_mem[c.item()].prototype
            if self.net.gpu:
                map(torch.cuda, [batch_p, old_p])
            new_p = self.momentum_update(old_p, batch_p, self.p_momentum)
            self.summarize_p_update(c.item(), new_p, old_p)

            class_mem[c.item()].prototype = new_p
        if self.net.log:
            print("* Updated Prototype *")
            print()
