#!/usr/bin/env python
import time
import numpy as np

import theano.tensor as T

from eca import ECA, Input, RegressionLayer
from utils import visualize, rect, MnistDataset


def run(dry_run=False):
    data = MnistDataset(batch_size=500,
                        testset_size=1000,
                        normalize=False,
                        as_one_hot=True)
    stiff_start, stiff_end, stiff_decay = (0.005, 0.005, 0.99)
    stiff_update = lambda s: s * stiff_decay + (1 - stiff_decay) * stiff_end
    input_dim = data.size('trn')[0][0]
    label_dim = data.size('trn')[1][0]

    class Model(ECA):
        def structure(self):
            if dry_run:
                self.U = Input('U', input_dim)
                self.Y = Input('Y', label_dim)
                RegressionLayer('Z', 3, (self.U, self.Y), rect,
                                min_tau=0.0, stiffx=1.0,
                                merge_op=lambda u, y: rect(u + y)),
            else:
                self.U = Input('U', input_dim)
                self.Y = Input('Y', label_dim)
                RegressionLayer('Z', 100, (self.U, self.Y), rect,
                                min_tau=0.0, stiffx=1.0,
                                merge_op=lambda u, y: u)
    mdl = Model()
    trn_iters = 10000 if not dry_run else 1

    #mdl.Y.nonlin_est = lambda x: T.nnet.softmax(x.T).T

    print 'Training', trn_iters, 'iterations'
    d = data.get('trn')
    stiff = stiff_start
    weights = []
    try:
        trn_sig = mdl.new_signals(data.samples('trn'))
        trne_sig = mdl.new_signals(data.samples('trn'))
        val_sig = mdl.new_signals(data.samples('val'))
        tst_sig = mdl.new_signals(data.samples('tst'))
        for i in range(1, trn_iters + 1):
            t = time.time()
            # Update model
            trn_sig.propagate(d.samples, d.labels)
            trn_sig.adapt_layers(stiff)

            stiff = stiff_update(stiff)
            if i % 200 == 0:
                calculate_accuracy(trne_sig, val_sig, tst_sig, data)

            # Progress prints
            if (i % 20 == 0):
                weights += [mdl.first_phi()[:-10, :]]
                i_str = "I %4d:" % (i)
                t_str = 't: %.2f s' % (time.time() - t)
                t = time.time()
                stiff_str = "stiff: %5.3f" % stiff

                tostr = lambda t: "{" + ", ".join(["%s: %6.2f" % (n, v) for (n, v) in t]) + "}"

                var_str = " logvar:" + tostr(trn_sig.variance())
                a_str   = " avg:   " + tostr(trn_sig.avg_levels())
                phi_norms = mdl.phi_norms()
                phi_larg_str = " |phinL|: " + tostr(map(lambda a: (a[0], np.sum(a[1] > 1.1)), phi_norms))
                phi_ones_str = " |phin1|: " + tostr(map(lambda a: (a[0], np.sum(np.isclose(a[1], 1.0, atol=0.1))), phi_norms))
                phi_zero_str = " |phin0|: " + tostr(map(lambda a: (a[0], np.sum(np.isclose(a[1], 0.0, atol=0.5))), phi_norms))
                phi_str = " |phi|: " + tostr(map(lambda a: (a[0], np.average(a[1])), phi_norms))
                E_str = " E: " + tostr(trn_sig.energy())
                y_acc = " yacc: %.2f" % (d.accuracy(trn_sig.y_est()) * 100.)
                Y_acc = " Yacc: %.2f" % (d.accuracy(trn_sig.Y.var.get_value()) * 100.)
                Y_cost = " Yerr: %.3f" % np.mean(trn_sig.Y.energy())
                U_cost = " Uerr: %.3f" % np.mean(trn_sig.U.energy())

                print i_str, stiff_str, t_str, y_acc, Y_acc, Y_cost, U_cost, phi_ones_str
                #print var_str, a_str, phi_str

    except KeyboardInterrupt:
        pass

    if dry_run:
        return

    try:
        print 'Calculating final accuracy'
        calculate_accuracy(trne_sig, val_sig, tst_sig, data)
        if False:
            visualize(weights[0])
    except KeyboardInterrupt:
        pass


def calculate_accuracy(trn_sig, val_sig, tst_sig, data):
    d = data.get('trn')
    y_est = trn_sig.converge(d.samples, 0.0 * d.labels).y_est()
    d.accuracy(y_est, print_it=True)

    y_est = trn_sig.converge(d.samples, np.float32(np.nan * d.labels)).y_est()
    d.accuracy(y_est, print_it=True)

    y_est = -trn_sig.converge(d.samples, np.float32(0.0 * d.labels)).Y.var.get_value()
    d.accuracy(y_est, print_it=True)

    d = data.get('val')
    y_est = -val_sig.converge(d.samples, np.zeros_like(d.labels)).Y.var.get_value()
    d.accuracy(y_est, print_it=True)

    y_est = val_sig.converge(d.samples, np.zeros_like(d.labels)).y_est()
    d.accuracy(y_est, print_it=True)

    y_est = val_sig.converge(d.samples, np.float32(np.nan * d.labels)).y_est()
    d.accuracy(y_est, print_it=True)


# TODO: Integrate with above
def run_with_visuals():
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    fig, axes = plt.subplots(2, 2)
    axes = self.axes = np.asarray(axes).flatten()
    fig.subplots_adjust(hspace=0.1)


    axes[0].set_title('phi_u')
    axes[1].set_title('trn(u, uest), tst(u, uest)')
    axes[2].set_title('generated digits 0 - 9')
    axes[3].set_title('trn(y, yest), tst(y, yest)')

    # Run one iteration to create state
    u, y = self.u, self.y = self.get_data('trn')
    self.stiff = self.stiff_start
    self.iter = 0

    self.run_iteration(self.iter, self.mdl, u, y, self.stiff)
    self.im = []
    for x in range(len(axes)):
        self.im += [axes[x].imshow(self.img(x),
                                   cmap=plt.get_cmap('gray'),
                                   interpolation='nearest')]

    ani = animation.FuncAnimation(fig, self.updatefig, interval=50, fargs=(self.mdl, u, y))
    self.iter = 0
    plt.show()

def updatefig(self, _, mdl, u, y, *args):
    for i in range(20):
        self.run_iteration(self.iter, mdl, u, y, self.stiff)
        self.stiff = self.stiff * self.stiff_alpha + (1 - self.stiff_alpha) * self.stiff_end
        self.iter += 1
    [self.im[x].set_array(self.img(x)) for x in range(len(self.im))]
    return self.im

def img(self, i):
    arr = rearrange_for_plot
    if i == 0:
        im = arr(self.mdl.first_phi())
    elif i == 1:
        im = np.vstack(
            (arr(self.u[:, :15]),
             arr(self.mdl.uest('training', no_eval=True)[:, :15].eval()),
             arr(self.get_data('val')[0][:, :15]),
             arr(self.mdl.uest('validation', no_eval=True)[:, :15].eval())))
    elif i == 2:
        im = arr(self.mdl.uest('u_gen'))
    elif i == 3:
        n = 50
        y_trn = np.fliplr(self.y[:, :n].T)
        space = np.zeros(y_trn.shape)
        trn = self.mdl.yest('training', no_eval=True)[:, :n].eval().T
        y_tst = np.fliplr(self.get_data('val')[1][:, :n].T)
        tst = self.mdl.yest('validation', no_eval=True)[:, :n].eval().T
        im = np.hstack((y_trn, trn, space, y_tst, tst))
    im += np.abs(np.min(im))
    im /= np.max(im)
    return im


def generate_u(self):
    uv, yv = self.get_data('val')

    uest = self.mdl.estimate_u(
        0.0 * uv[:, :10],
        np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]).T, 'u_gen')

    classify_by_reconstruction_error = False
    if classify_by_reconstruction_error:
        y_est = []
        for s in uv.T:
            cs = []
            for c in range(10):
                cs += [np.sum(np.square(s - uest[:, c]))]
            y_est += [-np.array(cs)]  # Minus because we take argmax later
        y_est = np.array(y_est).T
        print 'Training error by classifying based on reconstruction error',
        print self.accuracy(y_est, yv)

if __name__ == '__main__':
    run()
