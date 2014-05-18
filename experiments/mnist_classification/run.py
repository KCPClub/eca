#!/usr/bin/env python
import time
from utils import rearrange_for_plot
import numpy as np
import theano.tensor as T

from eca import ECA
from utils import MnistData


class TestCaseBase(object):
    def __init__(self):
        self.testset_size = 1000
        self.data = MnistData(batch_size=100, testset_size=self.testset_size)
        self.trn_iters = 400
        self.mdl = None
        self.tau_start = None
        self.tau_end = None
        self.tau_alpha = None

        self.configure()
        assert self.mdl is not None

        # This is global slowness that helps the model to get past the
        # unstable beginning..
        self.tau = self.tau_start
        self.iter = 0
        self.timestamp = time.time()
        (u, y) = self.get_data('trn')
        self.u, self.y = u, y

    def run(self):
        print 'Training...'
        try:
            for i in range(self.trn_iters):
                self.run_iteration()
            print 'Testing accuracy...'
            # TODO: test error broken at the moment, trust validation as it is not used for
            # hyper-parameter tuning at the moment
            #tst_acc = self.calculate_accuracy(ut, yt)
            tst_acc = np.nan
            print "Accuracy: Training %6.2f %%, validation %6.2f %%, test %6.2f %%" % (
                #100. * trn_acc,
                #100. * val_acc,
                np.nan, np.nan, 100. * tst_acc)

        except KeyboardInterrupt:
            pass

    def run_iteration(self):
        u, y = self.u, self.y
        t = time.time()
        self.mdl.update(u, y, self.tau)
        self.tau = self.tau * self.tau_alpha + (1 - self.tau_alpha) * self.tau_end

        i_str = "I %4d:" % (self.iter)
        # Progress prings
        if ((self.iter) % 20 == 0):
            t_str = 't: %.2f s' % (time.time() - t)
            t = time.time()
            tau_str = "Tau: %4.1f" % self.tau

            tostr = lambda t: "{" + ", ".join(["%s: %6.2f" % (n, v) for (n, v) in t]) + "}"

            var_str = " logvar:" + tostr(self.mdl.variance())
            a_str   = " avg:   " + tostr(self.mdl.avg_levels())
            phi_str = " |phi|: " + tostr(self.mdl.phi_norms())

            print i_str, tau_str, t_str, var_str, phi_str
            #print var_str, a_str, phi_str
            #print var_str, phi_str

        # Accuracy prints
        if ((self.iter) % 200 == 0):
            t2 = time.time()
            (trn_acc, val_acc) = self.calculate_accuracy(u, y)
            t_str = 't: %.2f s,' % (time.time() - t2)
            acc_str = "Accuracy trn %6.2f %%, val %6.2f %%" % (100. * trn_acc,
                                                               100. * val_acc)
            print i_str, acc_str, t_str
        self.iter += 1

    def configure(self):
        raise NotImplemented()

    def accuracy(self, y_est, y_true):
        y_true = y_true if len(y_true.shape) == 1 else np.argmax(y_true, axis=0)
        y_est = y_est if len(y_est.shape) == 1 else np.argmax(y_est, axis=0)
        return float(np.bincount(y_est == y_true, minlength=2)[1]) / len(y_est)

rect = lambda x: T.where(x < 0., 0., x)

class UnsupervisedLearning(TestCaseBase):
    def configure(self):
        layers = [20]  # Try e.g. [30, 20] for multiple layers and increase tau start
        self.tau_start = 30
        self.tau_end = 5
        self.tau_alpha = 0.99
        self.mdl = ECA(layers,
                       self.data.size('trn', 0)[0][0] + 10,
                       0,  # n of output
                       T.abs_)  # np.tanh, rect, None, etc..


    def get_data(self, type):
        assert type in ['tst', 'trn', 'val']

        (u, y) = self.data.get(type, i=0, as_one_hot=True)
        # Equalize energies
        u_avg_en = np.average(np.sum(np.square(u), axis=0))
        y_avg_en = np.average(np.sum(np.square(y), axis=0))
        #y *= np.sqrt(u_avg_en / y_avg_en)

        if type == 'trn':
            u = np.vstack([u, y])
        else:
            u = np.vstack((u, np.nan * np.zeros((10, u.shape[1]), dtype=np.float32)))
        return (u, y)

    def calculate_accuracy(self, u, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=0)

        # Training error
        y_est = self.mdl.uest()[-10:, :]
        trn_acc = self.accuracy(y_est[:, :self.testset_size],
                                y.T[:self.testset_size].T)

        # Validation error
        (uv, yv) = self.get_data('val')
        y_est = self.mdl.estimate_u(uv, None, 'validation')[-10:, :]
        val_acc = self.accuracy(y_est, yv)
        return (trn_acc, val_acc)

    def visualize(self):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        plt.imshow(rearrange_for_plot(self.mdl.first_phi()[:-10, :]), cmap=cm.Greys_r)
        plt.show()


class SupervisedLearning(TestCaseBase):
    def configure(self):
        layers = [100]  # Try e.g. [30, 20] for multiple layers and increase tau start
        self.tau_start = 20
        self.tau_end = 5
        self.tau_alpha = 0.99
        self.mdl = ECA(layers,
                       self.data.size('trn', 0)[0][0],
                       self.data.size('trn', 0, as_one_hot=True)[1][0],
                       #T.tanh)  # T.tanh, rect, None, etc..
                       rect)  # T.tanh, rect, None, etc..
                       #lambda x: x)

    def get_data(self, type):
        assert type in ['tst', 'trn', 'val']

        (u, y) = self.data.get(type, i=0, as_one_hot=True)
        #u, y = u.copy(), y.copy()
        ## Equalize energies to 1.0
        ##print np.var(u, axis=1)[100:110]
        #u_avg_en = np.average(np.sum(np.square(u), axis=0))
        #y_avg_en = np.average(np.sum(np.square(y), axis=0))
        #u /= np.sqrt(u_avg_en / y_avg_en)
        #y *= np.sqrt(u_avg_en / y_avg_en)
        #u_avg_en = np.average(np.sum(np.square(u), axis=0))
        #y_avg_en = np.average(np.sum(np.square(y), axis=0))
        #print np.max(y)
        #print np.var(u, axis=1)[100:110]
        #u *= 100.
        #y *= 100.
        #u_avg_en = np.average(np.sum(np.square(u), axis=0))
        #y_avg_en = np.average(np.sum(np.square(y), axis=0))
        #print u_avg_en, y_avg_en
        return (u, y)

    def calculate_accuracy(self, u, y):
        # Training error
        ut, yt = u[:, :self.testset_size], y.T[:self.testset_size].T
        trn_acc = self.accuracy(self.mdl.estimate_y(ut, np.nan * yt, 'training_err'), yt)

        # Validation error
        uv, yv = self.get_data('val')
        val_acc = self.accuracy(self.mdl.estimate_y(uv, np.nan * yv, 'validation'), yv)

        return (trn_acc, val_acc)

    def visualize(self):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        plt.imshow(rearrange_for_plot(self.mdl.first_phi()), cmap=cm.Greys_r)
        plt.show()


class SupervisedLearningGUI(SupervisedLearning):
    def run(self):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        fig, axes = plt.subplots(2, 2)
        axes = self.axes = np.asarray(axes).flatten()
        fig.subplots_adjust(hspace=0.1)

        # Run one iteration to create state
        self.run_iteration()

        axes[0].set_title('phi_u')
        axes[1].set_title('trn(u, uest), tst(u, uest)')
        axes[2].set_title('generated digits 0 - 9')
        axes[3].set_title('trn(y, yest), tst(y, yest)')
        self.im = []
        for x in range(len(axes)):
            self.im += [axes[x].imshow(self.img(x),
                                       cmap=plt.get_cmap('gray'),
                                       interpolation='nearest')]

        ani = animation.FuncAnimation(fig, self.updatefig, interval=50)
        plt.show()

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

    def updatefig(self, *args):
        for i in range(20):
            self.run_iteration()
        [self.im[x].set_array(self.img(x)) for x in range(len(self.im))]
        return self.im

    def calculate_accuracy(self, u, y):
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

        return super(SupervisedLearningGUI, self).calculate_accuracy(u, y)


def main():
    print 'Initializing...'
    #o = UnsupervisedLearning()
    #o = SupervisedLearningGUI()
    o = SupervisedLearning()
    o.run()

if __name__ == '__main__':
    main()
