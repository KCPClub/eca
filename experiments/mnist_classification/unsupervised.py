#!/usr/bin/env python
import time
import numpy as np

from eca import ECA, Input, Layer
from utils import visualize, rect, MnistDataset


def run(dry_run=False):
    data = MnistDataset(batch_size=100,
                        testset_size=1000,
                        normalize=False)
    stiff_start, stiff_end, stiff_decay = (0.5, 0.005, 0.90)
    stiff_update = lambda s: s * stiff_decay + (1 - stiff_decay) * stiff_end
    input_dim = data.size('trn')[0][0]

    class Model(ECA):
        def structure(self):
            if dry_run:
                self.U = Input('U', input_dim)
                self.X = Layer('X', 3, self.U, rect, 1.0)
            else:
                self.U = Input('U', input_dim)
                self.X = Layer('X1', 100, self.U, None, min_tau=0., stiffx=1.0)
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

            stiff = stiff_update(stiff)

            # Progress prints
            if (i % 20 == 0):
                weights += [mdl.first_phi()[:, :]]
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
