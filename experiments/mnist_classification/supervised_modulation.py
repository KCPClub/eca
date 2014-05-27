#!/usr/bin/env python
import time
import numpy as np

from eca import ECA
from utils import visualize, rect, MnistDataset


def run(dry_run=False):
    data = MnistDataset(batch_size=1000,
                        testset_size=1000,
                        normalize=False)
    stiff_start, stiff_end, stiff_decay = (0.5, 0.005, 0.99)
    stiff_update = lambda s: s * stiff_decay + (1 - stiff_decay) * stiff_end
    layers = [100] if not dry_run else [1]
    mdl = ECA(layers, data.size('trn', 0)[0][0], 0, rect)
    trn_iters = 500 if not dry_run else 1

    print 'Training...'
    d = data.get('trn')
    n = layers[0]
    weights = []
    try:
        # "Module model" so that it updates only certain weights at a time
        en = np.ones((n, d.samples.shape[1]), dtype=np.float32)
        for j in range(10):
            en[n//10 * j: n//10 * (j+1), d.labels != j] *= -0.1
        mdl.l_U.child.set_modulation(en)
        stiff = stiff_start

        for i in range(1, trn_iters + 1):
            t = time.time()
            # Update model
            mdl.update(d.samples, None, stiff)

            stiff = stiff_update(stiff)
            if i % 200 == 0:
                calculate_accuracy(mdl, data)

            # Progress prints
            if (i % 20 == 0):
                weights += [mdl.first_phi()[:-10, :]]
                i_str = "I %4d:" % (i)
                t_str = 't: %.2f s' % (time.time() - t)
                t = time.time()
                stiff_str = "stiff: %5.3f" % stiff

                tostr = lambda t: "{" + ", ".join(["%s: %6.2f" % (n, v) for (n, v) in t]) + "}"

                var_str = " logvar:" + tostr(mdl.variance())
                a_str   = " avg:   " + tostr(mdl.avg_levels())
                phi_norms = mdl.phi_norms()
                phi_larg_str = " |phinL|: " + tostr(map(lambda a: (a[0], np.sum(a[1] > 1.1)), phi_norms))
                phi_ones_str = " |phin1|: " + tostr(map(lambda a: (a[0], np.sum(np.isclose(a[1], 1.0, atol=0.1))), phi_norms))
                phi_zero_str = " |phin0|: " + tostr(map(lambda a: (a[0], np.sum(np.isclose(a[1], 0.0, atol=0.5))), phi_norms))
                phi_str = " |phi|: " + tostr(map(lambda a: (a[0], np.average(a[1])), phi_norms))
                E_str = " E: " + tostr(mdl.energy())

                print i_str, stiff_str, t_str, E_str, phi_ones_str, phi_zero_str, phi_larg_str
                #print var_str, a_str, phi_str

    except KeyboardInterrupt:
        pass

    if dry_run:
        return

    try:
        print 'Calculating final accuracy'
        calculate_accuracy(mdl, data)
        if False:
            visualize(weights[0])
    except KeyboardInterrupt:
        pass


def calculate_accuracy(mdl, data):
    n = mdl.l_U.child.n

    def calculate(type):
        d = data.get(type)
        x_en = np.square(mdl.estimate_x(d.samples, None, type))
        scores = []
        x_en /= np.mean(x_en, axis=1, keepdims=True)
        for i in range(10):
            en = np.mean(x_en[i * n//10:(i + 1) * n//10, :], axis=0)
            scores += [en]
        d.accuracy(np.array(scores), print_it=True)

    calculate('trn')
    calculate('val')

if __name__ == '__main__':
    run()
