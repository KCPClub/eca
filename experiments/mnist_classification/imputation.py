#!/usr/bin/env python
import time
import numpy as np

import theano.tensor as T

from eca import ECA, Input, Layer
from utils import visualize, rect, MnistDataset, imshowtiled
import matplotlib.pyplot as plt

CONTINUOUS_VISUALIZATION = False
SAVE_MOVIE = False


def impute(data, test_k, mdl, test_sig, guessed, true, sample=False):
    inp = guessed
    test_sig.converge(inp, None)

    # Plot the generated data
    fig, ax = plt.subplots(1, 5)
    fig.set_size_inches(15.5, 10.5)
    inplot = inp.copy()
    inplot[np.isnan(inp)] = 0.5
    u_est = test_sig.u_est(no_eval=True)
    if sample:
        u_est = T.sqr(u_est).eval()
        u_est -= np.minimum(0, np.min(u_est))
        u_est /= np.max(u_est) * 5
        guesses = np.random.binomial(n=1, p=u_est, size=(784, test_k))
        guessed = np.float32(np.where(np.isnan(guessed) & guesses == 1.0, guesses, guessed))
    else:
        u_est = u_est.eval()
    imshowtiled(mdl.first_phi(), axis=ax[0])
    imshowtiled(test_sig.U.var.get_value(), axis=ax[1])
    imshowtiled(u_est, axis=ax[2])
    imshowtiled(inplot, axis=ax[3])
    imshowtiled(true, axis=ax[4])
    plt.show()

    return guessed


def run(dry_run=False):
    trn_iters = 1000 if not dry_run else 1
    data = MnistDataset(batch_size=2000,
                        testset_size=1000,
                        normalize=False)
    stiff_start, stiff_end, stiff_decay = (0.5, 0.05, 0.90)
    stiff_update = lambda s: s * stiff_decay + (1 - stiff_decay) * stiff_end
    input_dim = data.size('trn')[0][0]

    class Model(ECA):
        def structure(self):
            if dry_run:
                self.U = Input('U', input_dim)
                self.X = Layer('X', 3, self.U, rect, 1.0)
            else:
                self.U = Input('U', input_dim)
                self.X1 = Layer('X1', 100, self.U, lambda x: rect(x), min_tau=0., stiffx=1.0)
    mdl = Model()

    print 'Training for', trn_iters, 'iterations'
    d = data.get('trn')
    k = data.samples('trn')
    stiff = stiff_start
    weights = []
    try:
        rng = np.random.RandomState(seed=0)
        trn_sig = mdl.new_signals(k)
        test_k = 25
        test_sig = mdl.new_signals(test_k)

        test_data = data.get('val').samples[:, :test_k]
        guessed = test_data.copy()

        RECONSTRUCT_HALF = False
        if RECONSTRUCT_HALF:
            guessed[:784//2] = np.nan
        else:
            mask = rng.binomial(n=1, p=0.1, size=(784, test_k))
            guessed = np.where(mask == 1, guessed, np.nan)

        for i in range(1, trn_iters + 1):
            t = time.time()
            # Training
            trn_sig.propagate(d.samples, None)
            trn_sig.adapt_layers(stiff)
            stiff = stiff_update(stiff)
            # Progress prints
            if (i % 20 == 0):
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

                impute(data, test_k, mdl, test_sig, guessed, test_data, sample=False)
                print i_str, stiff_str, t_str, E_str, phi_ones_str, u_err, U_sqr
                #print var_str, a_str, phi_str
    except KeyboardInterrupt:
        pass

    try:
        # Try to interpret u_est as pdf and sample from it
        while True and not dry_run:
            guessed = impute(data, test_k, mdl, test_sig, guessed, test_data, sample=True)
    except KeyboardInterrupt:
        pass

    try:
        if not dry_run and False:
            visualize(weights[0])
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    run()
