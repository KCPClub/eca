#!/usr/bin/env python
import time
import numpy as np

from eca import ECA
from utils import visualize, rect, MnistDataset


def run(dry_run=False):
    data = MnistDataset(batch_size=1000,
                        testset_size=1000,
                        normalize=False,
                        as_one_hot=True)
    stiff_start, stiff_end, stiff_decay = (0.5, 0.005, 0.99)
    stiff_update = lambda s: s * stiff_decay + (1 - stiff_decay) * stiff_end
    layers = [100] if not dry_run else [1]
    mdl = ECA(layers,
              data.size('trn', 0)[0][0],
              data.size('trn', 0)[1][0],
              rect)
    trn_iters = 500 if not dry_run else 1

    print 'Training...'
    d = data.get('trn')
    stiff = stiff_start
    weights = []
    try:
        for i in range(1, trn_iters + 1):
            t = time.time()
            # Update model
            mdl.update(d.samples, d.labels, stiff)

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
    d = data.get('trn')
    y_est = mdl.estimate_y(d.samples, np.float32(np.nan * d.labels), 'trn-err')
    d.accuracy(y_est, print_it=True)

    d = data.get('val')
    y_est = mdl.estimate_y(d.samples, np.float32(np.nan * d.labels), 'val-err')
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
