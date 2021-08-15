import importlib
import datetime
import argparse
import random
import uuid
import time
import os
import numpy as np
import torch

from metrics.metrics import confusion_matrix
import metrics.plot as plot
from metrics.utils import createdirs
from model.common import MLP, ResNet18
from model.prototypical.mem_scheme import MemoryScheme
from model.prototypical.loss_scheme import PPPloss
from model.prototypical.p_scheme import PrototypeScheme
from model.prototypical.sample_scheme import SampleScheme

parser = argparse.ArgumentParser(description='Continuum learning')

# experiment parameters
parser.add_argument('exp_name', default=None, type=str, help='id for the experiment.')
parser.add_argument('--cuda', type=str, default='yes', help='Use GPU?')
parser.add_argument('--iid', type=str, default='no', help='Make all tasks into 1 iid distr.')
parser.add_argument('--log_every', type=int, default=100, help='frequency of logs, in minibatches')
parser.add_argument('--save_path', type=str, default='results/', help='save models at the end of training')
parser.add_argument('--output_name', type=str, default='', help='special output name for the results?')
parser.add_argument('--n_seeds', default=5, type=int, help='Nb of seeds to run.')
parser.add_argument('--seed', default=None, type=int, help='Run a specific seed.')
parser.add_argument('--opt', default='sgd', type=str, choices=['sgd', 'adam'], help='Optimizer.')
parser.add_argument('--resume', default=None, type=str, help='resume in time/uid parentdir')

# CoPE Prototype schemes
parser.add_argument('--p_mode', default='batch_momentum_incr', type=str, choices=PrototypeScheme.valid_p_modes,
                    help='Update strategy for the class prototypes.')
parser.add_argument('--p_momentum', default=0.99, type=float,
                    help='Momentum of the moving avg updates for the prototypes.')

# CoPE Operational memory management
parser.add_argument('--qi_mode', default='reservoir', type=str, choices=MemoryScheme.valid_qi_modes,
                    help='Update strategy for the raw exemplars.')
parser.add_argument('--sample_qi_mode', default='rnd', type=str, choices=SampleScheme.valid_sample_qi_modes,
                    help='Sampling strategy for the raw exemplars.')
parser.add_argument('--dyn_mem', type=str, default='yes',
                    help='Use dynamic buffer allocation instead of a priori fixed class-based memory.')
# CoPE Loss
parser.add_argument('--loss_mode', default='joint', type=str, choices=PPPloss.modes,
                    help='PPP-Loss mode: Use only repellor (pos), attractor (neg) or standard both (joint)')
parser.add_argument('--loss_T', default=1, type=float, help='Softmax concentration level.')
parser.add_argument('--weight_decay', default=0, type=float, help='L2')
parser.add_argument('--momentum', default=0, type=float, help='Momentum in optimizer.')
parser.add_argument('--uid', default=None, type=str, help='id for the seed runs.')

# model parameters
parser.add_argument('--model', type=str, default='prototypical.CoPE',
                    choices=['prototypical.CoPE', 'finetune', 'reservoir', 'CoPE_CE', 'gem', 'icarl', 'GSSgreedy'],
                    help='model to train.')
parser.add_argument('--n_hiddens', type=int, default=100,
                    help='number of hidden neurons at each layer')
parser.add_argument('--n_layers', type=int, default=2,
                    help='number of hidden layers')
parser.add_argument('--shared_head', type=str, default='yes',
                    help='shared head between tasks')
parser.add_argument('--bias', type=int, default='1',
                    help='do we add bias to the last layer? does that cause problem?')
parser.add_argument('--n_outputs', type=int, default=None,
                    help='Define embedding size (def nb classes for CrossEntropy)')

# memory parameters
parser.add_argument('--n_memories', type=int, default=0,
                    help='number of input memories per task')
parser.add_argument('--n_sampled_memories', type=int, default=0,
                    help='number of sampled_memories per task')
parser.add_argument('--tasks_to_preserve', type=int, default=1,
                    help='max task to consider in the task sequence')
parser.add_argument('--normalize', type=str, default='no',
                    help='normalize gradients before selection')
parser.add_argument('--memory_strength', default=0, type=float,
                    help='memory strength (meaning depends on memory)')
parser.add_argument('--finetune', default='no', type=str,
                    help='whether to initialize nets in indep. nets')

# optimizer parameters
parser.add_argument('--n_epochs', type=int, default=1,
                    help='Number of epochs per task')
parser.add_argument('--n_iter', type=int, default=1,
                    help='Number of iterations per batch')
parser.add_argument('--batch_size', type=int, default=10,
                    help='batch size')
parser.add_argument('--mini_batch_size', type=int, default=0,
                    help='Subsample mini-batches from a batch for Single baseline.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='SGD learning rate')

# data parameters
parser.add_argument('--data_path', default='data/',
                    help='path where data is located')
parser.add_argument('--data_file', default='split_mnist.pt',
                    help='data file')
parser.add_argument('--samples_per_task', type=str, default='-1',
                    help='training samples per task (all if -1)\n'
                         'comma separated to define all task lengths, e.g. CIFAR10: 4000,400,400,400,400\n'
                         '|1,4000,400| to define T_i= Task 1 with 4000 samples, remaining tasks have 400')
parser.add_argument('--shuffle_tasks', type=str, default='no',
                    help='present tasks in order')
parser.add_argument('--eval_memory', type=str, default='no',
                    help='compute accuracy on memory')

# Featurespace plots (matplotlib dependency)
parser.add_argument('--visual', default=None, type=str,
                    help='Visualize data in feature space. Choose from tr/test/mem or split multiple by ",".')
parser.add_argument('--visual_chkpt', default='final', type=str, choices=['final', 'log'],
                    help='When to visualize. Final for final model, or at every log.')

# GSS
parser.add_argument('--subselect', type=int, default=1,
                    help='first subsample from recent memories')
parser.add_argument('--n_constraints', type=int, default=-1,
                    help='n_samples to replay from buffer for each new batch (paper: equal to batch size)')
parser.add_argument('--change_th', type=float, default=0.0,
                    help='gradients similarity change threshold for re-estimating the constraints')


# continuum iterator #########################################################
def load_datasets(args):
    print("path", args.data_path + '/' + args.data_file)
    d_tr, d_te = torch.load(args.data_path + '/' + args.data_file)
    n_inputs = d_tr[0][1].size(1)
    n_outputs = 0
    for i in range(len(d_tr)):
        n_outputs = max(n_outputs, d_tr[i][2].max().item())
        n_outputs = max(n_outputs, d_te[i][2].max().item())
    return d_tr, d_te, n_inputs, n_outputs + 1, len(d_tr)


class Continuum:
    def __init__(self, data, args):
        self.data = data
        self.batch_size = args.batch_size
        n_tasks = len(data)
        task_permutation = range(n_tasks)

        if args.shuffle_tasks == 'yes':
            task_permutation = torch.randperm(n_tasks).tolist()

        sample_idxs = []

        if '|' in args.samples_per_task:
            s_args = list(map(int, str(args.samples_per_task).replace('|', '').split(',')))
            assert len(s_args) == 3, \
                "Need (1)task number, (2)task length (3)other tasks length, got {}".format(s_args)
            samples_per_task = [s_args[2] for _ in range(n_tasks)]
            samples_per_task[int(s_args[0]) - 1] = s_args[1]
        else:
            samples_per_task = list(map(int, str(args.samples_per_task).split(",")))
        print("parsed samples_per_task={}".format(samples_per_task))

        # n = 1000
        for t in range(n_tasks):
            N = data[t][1].size(0)
            idx = t if len(samples_per_task) > t else 0
            if samples_per_task[idx] <= 0:
                n = N
            else:
                n = min(samples_per_task[idx], N)
            print("*********Task", t, "Samples are", n)
            p = torch.randperm(data[t][1].size(0))[0:n]
            sample_idxs.append(p)

        if args.iid:
            n_tasks = 1
            task_permutation = [0]

            min_class = np.inf
            max_class = -1
            x_tr = []
            y_tr = []
            for task_t, t_data in enumerate(
                    self.data):  # Each task like [(c1, c2), x_tr[i_tr].clone(), y_tr[i_tr].clone()]
                min_class = min(min_class, min(t_data[0]))  # (c1, c2)
                max_class = max(max_class, max(t_data[0]))  # (c1, c2)
                x_tr.extend(t_data[1][sample_idxs[task_t]])
                y_tr.extend(t_data[2][sample_idxs[task_t]])
            x_tr = torch.stack(x_tr, dim=0)
            y_tr = torch.stack(y_tr, dim=0)
            self.data = [[(min_class, max_class), x_tr, y_tr]]
            sample_idxs = [torch.randperm(y_tr.size(0))]

        self.task_idxs = []
        for t in range(n_tasks):
            task_t = task_permutation[t]

            for _ in range(args.n_epochs):
                task_p = [[task_t, i] for i in sample_idxs[task_t]]
                random.shuffle(task_p)
                self.task_idxs += task_p

        self.length = len(self.task_idxs)
        self.current = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.current >= self.length:
            raise StopIteration
        else:
            ti = self.task_idxs[self.current][0]
            j = []  # Idxs
            i = 0  # Count
            while (((self.current + i) < self.length) and
                   (self.task_idxs[self.current + i][0] == ti) and
                   (i < self.batch_size)):
                j.append(self.task_idxs[self.current + i][1])  # Take the 'batch-size' next idxs
                i += 1
            self.current += i
            j = torch.LongTensor(j)
            return self.data[ti][1][j], ti, self.data[ti][2][j]  # (x, t, y)


# train handle ###############################################################
def _eval_tasks(model, tasks, current_task, args):
    """
    Evaluates performance of the model on samples from all the tasks and reports
    1) average performance on all the samples regardless of their task.
    2) average performance up till current task.
    """
    model.eval()
    total_result_seq = []  # seq of isolated task accs
    total_size = 0
    total_pred = 0
    task_result_seq = []  # Snapshot running total for current task accuracy
    task_avg_acc = 0
    for t_idx, task in enumerate(tasks):
        x = task[1]
        y = task[2]
        task_correct = 0  # How many correctly predicted for this task

        eval_bs = x.size(0)
        for b_from in range(0, x.size(0), eval_bs):
            b_to = min(b_from + eval_bs, x.size(0) - 1)
            if b_from == b_to:
                xb = x[b_from].view(1, -1)
                yb = torch.LongTensor([y[b_to]]).view(1, -1)
            else:
                xb = x[b_from:b_to]
                yb = y[b_from:b_to]
            if args.cuda:
                xb = xb.cuda()
            _, pb = torch.max(model(xb, t_idx).data.cpu(), 1, keepdim=False)
            task_correct += (pb == yb).float().sum()  # How many correctly predicted

        task_acc = task_correct / x.size(0)  # Isolated task acc
        total_result_seq.append(task_acc)
        total_size += x.size(0)
        total_pred += task_correct

        # Snapshot running total for current task accuracy
        if t_idx == current_task:
            task_result_seq = [res for res in total_result_seq]
            task_avg_acc = total_pred / total_size

    # Total accuracy (further than current task)
    total_avg_acc = total_pred / total_size

    print("EVAL (train TASK {}/test total) ===> {}".format(current_task, total_result_seq))
    torch.save((model.state_dict(), task_result_seq, task_avg_acc), model.fname + '.pt')

    return total_result_seq, total_avg_acc, task_result_seq, task_avg_acc


def eval_tasks(model, tasks, current_task, args):
    """ No grads wrapper. """
    with torch.no_grad():
        return _eval_tasks(model, tasks, current_task, args)


def eval_on_memory(args, model):
    """ Compute accuracy on the buffer. """
    model.eval()
    acc_on_mem = 0
    if 'yes' in args.eval_memory:
        for x, y in zip(model.sampled_memory_data, model.sampled_memory_labs):
            if args.cuda:
                x = x.cuda()
            _, pb = torch.max(model(x.unsqueeze(0)).data.cpu(), 1, keepdim=False)
            acc_on_mem += (pb == y.data.cpu()).float()
        acc_on_mem = (acc_on_mem / model.sampled_memory_data.size(0))
    return acc_on_mem


class ResultTracker(object):

    def __init__(self):
        self.task_idxs = []  # Track for every log which task it belongs to
        self.tot_res_seqs = []  # per task accuracy up until the last task, Dim0= log_idx, Dim1= acc seq
        self.tot_avg_accs = []  # avg performance on all test samples
        self.task_res_seqs = []  # per task accuracy up until the current task
        self.task_avg_accs = []  # avg accuracy on task seen so far
        self.loss_history = {}

    def update(self, current_task, tot_res_seq, tot_avg_acc, task_res_seq, task_avg_acc):
        # Task-stamp
        self.task_idxs.append(current_task)
        # Total
        self.tot_res_seqs.append(tot_res_seq)
        self.tot_avg_accs.append(tot_avg_acc)
        # Task
        self.task_res_seqs.append(task_res_seq)
        self.task_avg_accs.append(task_avg_acc)

    def to_tensor(self):
        self.task_idxs = torch.Tensor(self.task_idxs)
        self.tot_res_seqs = torch.Tensor(self.tot_res_seqs)
        self.tot_avg_accs = torch.Tensor(self.tot_avg_accs)

    def get_all(self):
        return [self.task_idxs,
                self.tot_res_seqs,
                self.tot_avg_accs,
                self.task_res_seqs,
                self.task_avg_accs]


def life_experience(model, continuum, x_te, args):
    current_task = 0
    time_start = time.time()

    for (i, (x, t, y)) in enumerate(continuum):
        if t > args.tasks_to_preserve:
            print("Aborting: task exceeds task {}".format(args.tasks_to_preserve))
            break
        if (((i % args.log_every) == 0) or (t != current_task)):
            tot_res_seq, tot_avg_acc, task_res_seq, task_avg_acc = eval_tasks(model, x_te, current_task, args)
            args.tracker.update(current_task, tot_res_seq, tot_avg_acc, task_res_seq, task_avg_acc)

            if hasattr(model, "mem_update_scheme"):
                model.mem_update_scheme.print_mem_stats()
            if hasattr(model, "lossFunc"):
                model.lossFunc.tracker['log_it'].append(i)  # For loss tracking history
            model.log = True
            if args.visual and args.visual_chkpt == 'log':
                plot.plot_featspace(args.visual, continuum.data, x_te, model, current_task, i,
                                    save_img_path=args.imgname)
            current_task = t

        v_x = x.view(x.size(0), -1)
        v_y = y.long()

        if args.cuda:
            v_x = v_x.cuda()
            v_y = v_y.cuda()

        model.train()
        model.observe(v_x, t, v_y)
        model.log = False

    # Append final accs (after log_every)
    tot_res_seq, tot_avg_acc, task_res_seq, task_avg_acc = eval_tasks(model, x_te, args.tasks_to_preserve, args)
    args.tracker.update(current_task, tot_res_seq, tot_avg_acc, task_res_seq, task_avg_acc)
    args.tracker.to_tensor()

    if hasattr(model, "mem_update_scheme"):
        model.mem_update_scheme.print_mem_stats()
    if args.visual and args.visual_chkpt in ['log', 'final']:
        plot.plot_featspace(args.visual, continuum.data, x_te, model, current_task, "FINAL({})".format(len(continuum)),
                            save_img_path=args.imgname)

    # Get results on memories
    res_on_mem = eval_on_memory(args, model)

    time_end = time.time()
    time_spent = time_end - time_start

    return args.tracker, res_on_mem, time_spent


def get_model(args, n_inputs, n_outputs):
    nl, nh = args.n_layers, args.n_hiddens
    if args.is_cifar:
        net = ResNet18(n_outputs, bias=args.bias)
    else:
        net = MLP([n_inputs] + [nh] * nl + [n_outputs])
    return net


def main(overwrite_args=None):
    args = parser.parse_args()
    if overwrite_args is not None:
        for k, v in overwrite_args.items():  # Debugging
            setattr(args, k, v)

    args.dyn_mem = True if args.dyn_mem == 'yes' else False
    args.cuda = True if args.cuda == 'yes' else False
    args.finetune = True if args.finetune == 'yes' else False
    args.normalize = True if args.normalize == 'yes' else False
    args.shared_head = True if args.shared_head == 'yes' else False
    args.iid = True if args.iid == 'yes' else False

    if args.mini_batch_size == 0:
        args.mini_batch_size = args.batch_size  # no mini iterations

    # unique identifier
    uid = uuid.uuid4().hex if args.uid is None else args.uid
    now = str(datetime.datetime.now().date()) + "_" + ':'.join(str(datetime.datetime.now().time()).split(':')[:-1])
    runname = 'T={}_id={}'.format(now, uid) if not args.resume else args.resume

    # Paths
    setupname = [args.exp_name, args.model, args.data_file.split('.')[0]]
    parentdir = os.path.join(args.save_path, '_'.join(setupname))

    print("Init args={}".format(args))
    stat_files = []
    seeds = [args.seed] if args.seed is not None else list(range(args.n_seeds))
    for seed in seeds:
        # initialize seeds
        print("STARTING SEED {}/{}".format(seed, args.n_seeds - 1))
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if args.cuda:
            torch.cuda.manual_seed_all(seed)

        # load data
        x_tr, x_te, n_inputs, n_classes, n_tasks = load_datasets(args)
        args.is_cifar = ('cifar10' in args.data_file)
        args.is_mnist = ('mnist' in args.data_file)
        assert not (args.is_cifar and args.is_mnist)

        args.input_shape = x_tr[0][1][0].shape
        if args.input_shape[-1] == 3072:  # CIFAR
            assert args.is_cifar
            args.CHW = (3, 32, 32)
        elif args.input_shape[-1] == 784:  # MNIST
            assert args.is_mnist
            args.CHW = (1, 28, 28)
        else:
            raise NotImplementedError()

        args.n_classes = n_classes
        n_outputs = args.n_classes if args.n_outputs is None else args.n_outputs  # Embedding or Softmax

        # set up continuum
        continuum = Continuum(x_tr, args)

        # load model
        args.tracker = ResultTracker()
        args.net = get_model(args, n_inputs, n_outputs)
        Model = importlib.import_module('model.' + args.model)
        model = Model.Net(n_inputs, n_outputs, n_tasks, args)

        # set up file name for saving/chkpt
        if args.n_sampled_memories == 0:
            args.n_sampled_memories = args.n_memories
        if args.output_name:
            model.fname = args.output_name
        model.fname = os.path.join(parentdir, runname, 'seed={}'.format(seed))
        args.imgname = os.path.join('./img', '_'.join(setupname), '{}_{}/'.format(runname, 'seed={}'.format(seed)))

        if os.path.isfile(model.fname + '.pt'):
            print("[CHECKPOINT] Loading seed checkpoint: {}".format(model.fname + '.pt'))
            chkpt = torch.load(model.fname + '.pt')
            if hasattr(chkpt[-1], 'output_name'):  # See if is args object
                args = chkpt[-1]
                stat_files.append(model.fname + '.pt')  # For final accs
                print("Args overwritten by chkpt: {}".format(args))
                continue
            print("Checkpoint not restored, continuing with args: {}".format(args))

        createdirs(model.fname)
        createdirs(args.imgname)

        # prepare saving path and file name
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        if args.cuda:
            model.cuda()

        # run model on continuum
        res, acc_on_mem, spent_time = life_experience(model, continuum, x_te, args)

        # save confusion matrix and print one line of stats
        stats = confusion_matrix(res.task_idxs, res.tot_res_seqs, res.tot_avg_accs, acc_on_mem, args.tasks_to_preserve,
                                 model.fname + '.txt')
        one_liner = str(vars(args)) + ' # '
        one_liner += ' '.join(["%.3f" % stat for stat in stats])
        print(model.fname + ': ' + one_liner + ' # ' + str(spent_time))

        # save all results in binary file
        torch.save((*res.get_all(), model.state_dict(), stats, one_liner, args), model.fname + '.pt')
        stat_files.append(model.fname + '.pt')

    mean, std = stat_summarize(stat_files)
    print("FINISHED SCRIPT")


def stat_summarize(stat_files):
    print("Taking avg of {} results: {}".format(len(stat_files), stat_files))
    res = [torch.load(x) for x in stat_files]

    # Acc
    avg_acc = [x[6][0].unsqueeze(0) for x in res]
    print("Avg accs={}".format(avg_acc))
    avg_acc_t = torch.cat(avg_acc)
    mean = avg_acc_t.mean() * 100
    std = avg_acc_t.std() * 100
    print("Avg acc = {:.3f}+-{:.3f}".format(mean, std))

    return mean, std


if __name__ == "__main__":
    main()
