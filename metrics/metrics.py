# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import torch


def task_changes(result_t):
    n_tasks = int(result_t.max() + 1)
    changes = []
    current = result_t[0]
    for i, t in enumerate(result_t):
        if t != current:
            changes.append(i)
            current = t

    return n_tasks, changes


def confusion_matrix(task_idxs, result_seqs, avg_accs, acc_on_mem, max_task=0, fname=None):
    """
    """
    nt, changes = task_changes(task_idxs)
    changes = torch.LongTensor(changes + [result_seqs.size(0)]) - 1

    baseline = result_seqs[0]  # First performance measured on all tasks
    result = result_seqs[changes]  # Only keep at task changes

    # acc[t] equals result[t,t]
    acc = result.diag()  # Results at the end of the tasks
    fin = result[nt - 1][:max_task + 1]  # Results at the end of the sequence

    # bwt[t] equals result[T,t] - acc[t]
    bwt = result[nt - 1][:max_task + 1] - acc

    # fwt[t] equals result[t-1,t] - baseline[t]
    fwt = torch.zeros(nt)
    for t in range(1, nt):
        fwt[t] = result[t - 1, t] - baseline[t]

    if fname is not None:
        f = open(fname, 'w')

        print(' '.join(['%.4f' % r for r in baseline]), file=f)
        print('|', file=f)
        for row in range(result.size(0)):
            print(' '.join(['%.4f' % r for r in result[row]]), file=f)
        print('', file=f)
        print('Final Accuracy: %.4f' % fin.mean(), file=f)
        print('Backward: %.4f' % bwt.mean(), file=f)
        print('Forward:  %.4f' % fwt.mean(), file=f)
        print('Average Accuracy:  %.4f' % avg_accs[-1], file=f)
        print('Memory Accuracy:  %.4f' % acc_on_mem, file=f)
        print("Wrote results to file '{}'".format(fname))
        f.close()

    stats = []
    stats.append(fin.mean())
    stats.append(bwt.mean())
    stats.append(fwt.mean())
    return stats
