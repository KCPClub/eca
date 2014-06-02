#!/usr/bin/env python
import time
import numpy as np

import theano.tensor as T

from eca import ECA, Input, Layer
from utils import visualize, rect, MnistDataset, rearrange_for_plot

CONTINUOUS_VISUALIZATION = False
SAVE_MOVIE = False


def run(dry_run=False):
    data = MnistDataset(batch_size=100,
                        testset_size=1000,
                        normalize=False)
    stiff_start, stiff_end, stiff_decay = (0.05, 0.05, 0.90)
    stiff_update = lambda s: s * stiff_decay + (1 - stiff_decay) * stiff_end
    input_dim = data.size('trn')[0][0]

    class Model(ECA):
        def structure(self):
            if dry_run:
                self.U = Input('U', input_dim)
                self.X = Layer('X', 3, self.U, rect, 1.0)
            else:
                self.U = Input('U', input_dim)
                self.X1 = Layer('X1', 20, self.U, rect, min_tau=0., stiffx=1.0)
                self.X2 = Layer('X2',  5, self.X1, rect, min_tau=0., stiffx=1.0)
                self.X2.disable()
    mdl = Model()
    trn_iters = 1000 if not dry_run else 1

    print 'Training', trn_iters, 'iterations'
    d = data.get('trn')
    stiff = stiff_start
    weights = []
    try:
        trn_sig = mdl.new_signals(data.samples('trn'))
        for i in range(1, trn_iters + 1):
            t = time.time()
            # Update model
            trn_sig.propagate(d.samples, None)
            trn_sig.adapt_layers(stiff)

            # Enable second layer once the first layer has settled
            if i == 250:
                mdl.X2.enable()

            stiff = stiff_update(stiff)

            if CONTINUOUS_VISUALIZATION and i % (1 if SAVE_MOVIE else 5) == 0:
                import matplotlib.pyplot as plt
                if SAVE_MOVIE and i == 1:
                    import matplotlib.animation as animation
                    fig, ax = plt.subplots(2, 2)
                    fig.set_size_inches(10.5, 10.5)
                    writer = animation.writers['ffmpeg'](fps=10, bitrate=1000)
                    writer.setup(fig, 'eca-multilayer-rec.mp4', dpi=100)
                elif not SAVE_MOVIE:
                    fig, ax = plt.subplots(2, 2)
                    fig.set_size_inches(10.5, 10.5)

                u, U, X, phi, Q, = (d.samples,
                                    trn_sig.U.var.get_value(),
                                    trn_sig.U.next.var.get_value(),
                                    mdl.first_phi(),
                                    mdl.U.next.Q.get_value())
                X2 = trn_sig.U.next.next.var.get_value()

                import matplotlib.cm as cm
                n = 50
                svd = lambda x: np.linalg.svd(x, compute_uv=False)
                order = np.argsort(svd(X))
                pu, = ax[0, 0].plot(svd(u)[order][:-n:-1], 'o-')
                pU, = ax[0, 0].plot(svd(U)[order][:-n:-1], 'v-')
                pX, = ax[0, 0].plot(svd(X)[order][:-n:-1], 'x-')
                pX2, = ax[0, 0].plot(np.sort(svd(X2))[:-n:-1], 'x-')
                ax[0, 0].legend([pu, pU, pX, pX2], ['svd(u)', 'svd(U)', 'svd(X)', 'svd(X2)'])

                XE = np.mean(np.square(X), axis=1)
                #XE = np.var(X, axis=1)
                nphi = np.linalg.norm(phi.T, axis=1)
                order = np.arange(len(XE))
                pXE, = ax[0, 1].plot(XE[order][:-n:-1], 's-')
                pphi, = ax[0, 1].plot(nphi[order][:-n:-1], '*-')
                pq, = ax[0, 1].plot(Q[order, order][:-n:-1], 'x-')
                ax[0, 1].legend([pXE, pphi, pq], ['E{X^2}', '|phi|', 'q_i'])
                ax[0, 1].set_ylim([0.2, 5])

                XE2 = np.mean(np.square(X2), axis=1)
                nphi2 = np.linalg.norm(mdl.X2.phi.get_value().T, axis=1)

                order = np.arange(len(XE2))
                Q2 = mdl.X2.Q.get_value()
                pXE2, = ax[1, 0].plot(XE2[order][:-n:-1], 's-')
                pphi2, = ax[1, 0].plot(nphi2[order][:-n:-1], '*-')
                pq2, = ax[1, 0].plot(Q2[order, order][:-n:-1], 'x-')
                ax[1, 0].legend([pXE2, pphi2, pq2], ['E{X2^2}', '|phi2|', 'q2_i'])
                ax[1, 0].set_ylim([0.2, 3])

                ax[1, 1].imshow(rearrange_for_plot(phi), cmap=cm.Greys_r)
                if SAVE_MOVIE:
                    if i < 500:
                        writer.grab_frame()
                        ax[0, 0].clear()
                        ax[0, 1].clear()
                        ax[1, 0].clear()
                    elif i == 500:
                        writer.finish()
                else:
                    fig = plt.gcf()
                    plt.show()

            # Progress prints
            if (i % 20 == 0):
                weights += [mdl.first_phi()[:, :]]
                if not CONTINUOUS_VISUALIZATION:
                    visualize(weights[-1])
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
                u_err = ' uerr: %.5f' % trn_sig.u_err(d.samples)
                U_sqr = ' Usqr: %.5f' % np.average(np.square(trn_sig.U.var.get_value()))

                print i_str, stiff_str, t_str, E_str, phi_ones_str, u_err, U_sqr
                #print var_str, a_str, phi_str


    except KeyboardInterrupt:
        pass

    try:
        if not dry_run and False:
            visualize(weights[0])
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    run()
