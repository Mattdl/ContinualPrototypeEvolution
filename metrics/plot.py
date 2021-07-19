import numpy as np
import torch
import os
from metrics.utils import createdirs


def plot_featspace(data_mode, tr_data, eval_data, model, plot_t, batch_cnt, save_img_path=None, legend=False):
    modes = data_mode.split(',')
    init = 'pca'  # 'random'
    for mode in modes:
        mode_img_path = None
        if save_img_path is not None:
            mode_img_path = os.path.join(save_img_path, mode + "/")
            createdirs(mode_img_path)
        _ = _plot_featspace(mode, tr_data, eval_data, model, plot_t, batch_cnt, mode_img_path, legend, init,
                            figsize=(8, 8))
        _ = _plot_featspace(mode, tr_data, eval_data, model, plot_t, batch_cnt, mode_img_path, legend, init,
                            figsize=(10, 6))


def _plot(d_x2d, d_y, d_yunique, p_x2d, p_y, data_mode, plot_t, batch_cnt, legend, save_img_path, figsize=(8, 8)):
    from matplotlib import pyplot as plt
    import matplotlib
    from matplotlib import rcParams

    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    rcParams['font.family'] = 'DejaVu Serif'
    rcParams['font.sans-serif'] = ['DejaVuSerif']

    # PLOT
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')  # No borders/nbs

    # plt.figure()
    labels = ['Class {}'.format(i) for i in d_yunique]
    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'][:len(labels)]
    colors = ['C{}'.format(i) for i in range(10)][:len(labels)]
    plots = []
    for y, col, label in zip(d_yunique, colors, labels):
        plot = plt.scatter(d_x2d[d_y == y, 0], d_x2d[d_y == y, 1], c=col, label=label, marker='.', alpha=0.3)
        plots.append(plot)

    # ADD prototypes in different color/marker
    if len(p_y) > 0:
        pcol = 'black'
        pmarker = 'o'
        plt.scatter(p_x2d[:, 0], p_x2d[:, 1], c=pcol, marker=pmarker)

        for i in range(len(p_y)):
            ax.annotate(r'${\bf p}^' + '{}$'.format(str(p_y[i])), xy=(p_x2d[i, 0], p_x2d[i, 1]),
                        xytext=(-10, 14),
                        textcoords='offset points',  # ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
                        size=22
                        )

    # LEGEND (exclude p)
    if legend:
        lgnd = plt.legend(plots, labels,
                          loc='center right', bbox_to_anchor=(1.5, 0.5),
                          fontsize=22,
                          prop={'size': 20},
                          )

        for hndlr in lgnd.legendHandles:
            hndlr.set_alpha(1)
            hndlr.set_sizes([80])  # Dot in the legend

    # SAVE/SHOW PLOT
    plt.tight_layout()

    if save_img_path is not None:
        import datetime
        now = str(datetime.datetime.now().date()) + "_" + ':'.join(str(datetime.datetime.now().time()).split(':')[:-1])
        name = '{}_Task{}_batchcnt={}_{}'.format(data_mode, plot_t, batch_cnt, now)
        # Save 3 formats
        ext = '.pdf'
        savename = os.path.join(save_img_path, name + ext)
        plt.savefig(savename, bbox_inches='tight')

        ext = '.png'
        savename = os.path.join(save_img_path, name + ext)
        plt.savefig(savename, bbox_inches='tight')

        plt.clf()
        print("* SAVED FIG: {}".format(savename))
    else:
        plt.show()


def _plot_featspace(data_mode, tr_data, eval_data, model, plot_t, batch_cnt, save_img_path, legend, init,
                    figsize=(8, 8)):
    ret_init = init
    with torch.no_grad():
        from sklearn.manifold import TSNE

        # ADD DATA POINTS IN FEAT SPACE
        def get_data():
            data = tr_data[:plot_t + 1] if data_mode == 'tr' else eval_data[:plot_t + 1]
            d_x = torch.cat([t[1] for t in data], dim=0).cuda()  # [(),x,y] entry per task
            d_y = torch.cat([t[2] for t in data], dim=0)
            d_yunique = torch.unique(d_y).squeeze()
            return d_x, d_y.cpu().numpy(), d_yunique.cpu().numpy()

        def get_mem_data():
            d_x, d_y, d_yunique = [], [], []
            if len(model.class_mem) == 0:
                print("* Nothing in memory to plot")
                return
            for c, cmem in model.class_mem.items():
                d_x.append(cmem.qi)
                d_y.extend([c] * len(cmem.qi))
                d_yunique.append(c)
            d_x = torch.cat(d_x, dim=0).cuda()
            d_y = np.array(d_y)
            d_yunique = np.array(d_yunique)
            return d_x, d_y, d_yunique

        if data_mode == 'test_mem' and len(model.class_mem) == 0:
            data_mode = 'test'

        if data_mode == 'tr' or data_mode == 'test':
            d_x, d_y, d_yunique = get_data()
        elif data_mode == 'mem':  # Model class_mem
            d_x, d_y, d_yunique = get_mem_data()
        elif data_mode == 'test_mem':  # test/mem in same tsne projection
            d_x1, d_y1, d_yunique1 = get_data()
            d_x2, d_y2, d_yunique2 = get_mem_data()

            d_x = torch.cat([d_x1, d_x2], dim=0).cuda()
            d_y = np.concatenate((d_y1, d_y2), axis=0)
            d_yunique = np.concatenate((d_yunique1, d_yunique2), axis=0)
        else:
            raise NotImplementedError()

        # Forward
        f_x = model.forward_training(d_x)
        tsne_in = f_x.cpu().numpy()

        # ADD PROTOTOYPES
        p_x = []
        p_y = []
        for c, cmem in model.class_mem.items():
            p_x.append(cmem.prototype)
            p_y.append(c)
        if len(p_y) > 0:
            p_x = torch.cat(p_x, dim=0).cpu().numpy()
            tsne_in = np.concatenate((tsne_in, p_x), axis=0)

        # PROJECT
        tsne = TSNE(n_components=2, random_state=0,
                    init=init)  # Init kw can be used to init with prev values! (for timelapse)
        x2d = tsne.fit_transform(tsne_in)
        d_x2d = x2d[:d_x.shape[0]]
        p_x2d = x2d[d_x.shape[0]:]

        if data_mode == 'test_mem':
            d_x2d1 = d_x2d[:len(d_y1)]
            d_x2d2 = d_x2d[len(d_y1):]
            dmode = 'test_mem[TEST]'
            _plot(d_x2d1, d_y1, d_yunique, p_x2d, p_y, dmode, plot_t, batch_cnt, legend, save_img_path, figsize)

            dmode = 'test_mem[MEM]'
            _plot(d_x2d2, d_y2, d_yunique, p_x2d, p_y, dmode, plot_t, batch_cnt, legend, save_img_path, figsize)
        else:
            _plot(d_x2d, d_y, d_yunique, p_x2d, p_y, data_mode, plot_t, batch_cnt, legend, save_img_path, figsize)

        if len(p_y) > 0:
            ret_init = p_x2d

        return ret_init
