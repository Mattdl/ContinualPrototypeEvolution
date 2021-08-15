import torch
import traceback


class PPPloss(object):
    """ Pseudo-Prototypical Proxy loss
    """
    modes = ["joint", "pos", "neg"]  # Include attractor (pos) or repellor (neg) terms

    def __init__(self, net, mode="joint", T=1, tracker=None, ):
        """
        :param margin: margin on distance between pos vs neg samples (see TripletMarginLoss)
        :param dist: distance function 2 vectors (e.g. L2-norm, CosineSimilarity,...)
        """
        assert mode in self.modes
        self.net = net
        self.mode = mode
        self.T = T
        self.margin = 1

        # INIT tracker
        self.tracker = tracker
        self.tracker['log_it'] = []
        self.tracker['loss'] = []
        self.tracker['lnL_pos'] = []
        self.tracker['lnL_neg'] = []

    def __call__(self, x_metric, labels, model, eps=1e-8):
        """
        Standard reduction is mean, as we use full batch information instead of per-sample.
        Symmetry in the distance function inhibits summing loss over all samples.

        :param x_metric: embedding output of batch
        :param labels: labels batch
        :param class_mem: Stored prototypes/exemplars per seen class
        """
        if self.mode == "joint":
            pos, neg = True, True
        elif self.mode == "pos":
            pos, neg = True, False
        elif self.mode == "neg":
            pos, neg = False, True
        else:
            raise NotImplementedError()
        return self.softmax_joint(x_metric, labels, gpu=model.gpu, pos=pos, neg=neg)

    def softmax_joint(self, x_metric, y, gpu=True, pos=True, neg=True):
        """
        - \sum_{i in B^c} log(Pc) - \sum_{i in B^c}  \sum_{k \ne c} log( (1 - Pk))

        Note:
        log(Exp(y)) makes y always positive, which is required for our loss.
        """
        if torch.isnan(x_metric).any():
            print("skipping NaN batch")
            return torch.tensor(0)
        assert pos or neg, "At least one of the pos/neg terms must be activated in the Loss!"
        assert len(x_metric.shape) == 2, "Should only have batch and metric dimension."
        bs = x_metric.size(0)

        # All prototypes
        p_x, p_y = self.net.get_all_prototypes()

        # Init
        loss = None
        y_unique = torch.unique(y).squeeze()
        neg = False if len(y_unique.size()) == 0 else neg  # If only from the same class, there is no neg term
        y_unique = y_unique.view(-1)

        # Log
        tmplate = str("{: >20} " * y_unique.size(0))
        if self.net.log:
            print("\n".join(["-" * 40, "LOSS", tmplate.format(*list(y_unique))]))
            self.tracker['lnL_pos'].append(0)
            self.tracker['lnL_neg'].append(0)

        for label_idx in range(y_unique.size(0)):  # [summation over i]
            c = y_unique[label_idx]

            # Select from batch
            xc_idxs = (y == c).nonzero().squeeze(dim=1)
            xc = x_metric.index_select(0, xc_idxs)

            xk_idxs = (y != c).nonzero().squeeze(dim=1)
            xk = x_metric.index_select(0, xk_idxs)

            p_idx = (p_y == c).nonzero().squeeze(dim=1)
            pc = p_x[p_idx].detach()
            pk = torch.cat([p_x[:p_idx], p_x[p_idx + 1:]]).detach()  # Other class prototypes
            if self.net.log:
                print("Class {}:".format(str(c.item())), end='')

            lnL_pos = self.attractor(pc, pk, xc, gpu, include_batch=True) if pos else 0  # Pos
            lnL_neg = self.repellor(pc, pk, xc, xk, gpu, include_batch=True) if neg else 0  # Neg

            # Pos + Neg
            Loss_c = -lnL_pos - lnL_neg  # - \sum_{i in B^c} log(Pc) - \sum_{i in B^c}  \sum_{k \ne c} log( (1 - Pk))
            if self.net.log:
                print("{: >20}".format(
                    "| TOTAL: {:.1f} + {:.1f} = {:.1f}".format(float(-lnL_pos), float(-lnL_neg), float(Loss_c))))
                self.tracker['lnL_pos'][-1] -= lnL_pos.item()
                self.tracker['lnL_neg'][-1] -= lnL_neg.item()

            # Update loss
            loss = Loss_c if loss is None else loss + Loss_c

            # Checks
            try:
                assert lnL_pos <= 0
                assert lnL_neg <= 0
                assert loss >= 0 and loss < 1e10
            except:
                traceback.print_exc()
                exit(1)
        if self.net.log:
            self.tracker['loss'].append(loss.item())
            print("-" * 40)
        return loss / bs  # Make independent batch size

    def repellor(self, pc, pk, xc, xk, gpu, include_batch=True):
        # Gather per other-class samples
        if not include_batch:
            union_c = pc
        else:
            union_c = torch.cat([xc, pc])
        union_ck = torch.cat([union_c, pk]) #.clone().detach()
        c_split = union_c.shape[0]
        if gpu:
            union_ck = union_ck.cuda()

        neg_Lterms = torch.mm(union_ck, xk.t()).div_(self.T).exp_()  # Last row is with own prototype
        pk_terms = neg_Lterms[c_split:].sum(dim=0).unsqueeze(0)  # For normalization
        pc_terms = neg_Lterms[:c_split]
        Pneg = pc_terms / (pc_terms + pk_terms)

        expPneg = (Pneg[:-1] + Pneg[-1].unsqueeze(0)) / 2  # Expectation pseudo/prototype
        lnPneg_k = expPneg.mul_(-1).add_(1).log_()  # log( (1 - Pk))
        lnPneg = lnPneg_k.sum()  # Sum over (pseudo-prototypes), and instances

        if self.net.log:
            print(" + (#k) {:.1f}/({:.1f} + {:.1f})".format(float(pc_terms.mean().item()),
                                                            float(pc_terms.mean().item()),
                                                            float(pk_terms.mean().item())), end='')
        try:
            assert -10e10 < lnPneg <= 0
        except:
            print("error")
            traceback.print_exc()
            exit(1)
        return lnPneg

    def attractor(self, pc, pk, xc, gpu, include_batch=True):
        # Union: Current class batch-instances, prototype, memory
        if include_batch:
            pos_union_l = [xc.clone()]
            pos_len = xc.shape[0]
        else:
            pos_union_l = []
            pos_len = 1
        pos_union_l.append(pc)

        if gpu:
            pos_union_l = [x.cuda() for x in pos_union_l]
        pos_union = torch.cat(pos_union_l)
        all_pos_union = torch.cat([pos_union, pk]).clone().detach()  # Include all other-class prototypes p_k
        pk_offset = pos_union.shape[0]  # from when starts p_k

        # Resulting distance columns are per-instance loss terms (don't include self => diagonal)
        pos_Lterms = torch.mm(all_pos_union, xc.t()).div_(self.T).exp_()  # .fill_diagonal_(0)
        if include_batch:
            mask = torch.eye(*pos_Lterms.shape).bool().cuda() if gpu else torch.eye(*pos_Lterms.shape).bool()
            pos_Lterms = pos_Lterms.masked_fill(mask, 0)  # Fill with zeros

        Lc_pos = pos_Lterms[:pk_offset]
        Lk_pos = pos_Lterms[pk_offset:].sum(dim=0)  # sum column dist to pk's

        # Divide each of the terms by itself+ Lk term to get probability
        Pc_pos = Lc_pos / (Lc_pos + Lk_pos)
        expPc_pos = Pc_pos.sum(0) / (pos_len)  # Don't count self in
        lnL_pos = expPc_pos.log_().sum()

        # Sum instance loss-terms (per-column), divide by pk distances as well
        if self.net.log:
            print("(#c) {:.1f}/({:.1f} + {:.1f}) ".format(
                float(Lc_pos.mean().item()), float(Lc_pos.mean().item()), float(Lk_pos.mean().item())),
                end='')
        try:
            assert lnL_pos <= 0
        except:
            traceback.print_exc()
            exit(1)
        return lnL_pos
