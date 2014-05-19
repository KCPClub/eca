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
        self.data = MnistData(batch_size=1000, testset_size=self.testset_size)
        self.trn_iters = 410
        self.mdl = None
        self.tau_start = None
        self.tau_end = None
        self.tau_alpha = None
        self.tau_update = lambda tau: tau * self.tau_alpha + (1 - self.tau_alpha) * self.tau_end

        self.configure()
        assert self.mdl is not None

    def run(self):
        print 'Training...'
        (u, y) = self.get_data('trn')
        tau = self.tau_start
        self.iter = 0
        try:
            for i in range(self.trn_iters):
                self.run_iteration(i, self.mdl, u, y, tau)
                tau = self.tau_update(tau)
                if self.iter % 200 == 0:
                    self.print_accuracy(self.iter)
                self.iter += 1
            print 'Testing accuracy...'
            # TODO: test error broken at the moment, trust validation as it is not used for
            # hyper-parameter tuning at the moment
            #tst_acc = self.calculate_accuracy()
            tst_acc = np.nan
            print "Accuracy: Training %6.2f %%, validation %6.2f %%, test %6.2f %%" % (
                #100. * trn_acc,
                #100. * val_acc,
                np.nan, np.nan, 100. * tst_acc)

        except KeyboardInterrupt:
            pass

    def run_iteration(self, i, mdl, u, y, tau):
        t = time.time()
        mdl.update(u, y, tau)

        # Progress prings
        if ((i) % 20 == 0):
            i_str = "I %4d:" % (i)
            t_str = 't: %.2f s' % (time.time() - t)
            t = time.time()
            tau_str = "Tau: %4.1f" % tau

            tostr = lambda t: "{" + ", ".join(["%s: %6.2f" % (n, v) for (n, v) in t]) + "}"

            var_str = " logvar:" + tostr(mdl.variance())
            a_str   = " avg:   " + tostr(mdl.avg_levels())
            phi_str = " |phi|: " + tostr(mdl.phi_norms())

            print i_str, tau_str, t_str, var_str, phi_str
            #print var_str, a_str, phi_str
            #print var_str, phi_str


    def print_accuracy(self, i):
        i_str = "I %4d:" % (i)
        # Accuracy prints
        t2 = time.time()
        (trn_acc, val_acc) = self.calculate_accuracy()
        t_str = 't: %.2f s,' % (time.time() - t2)
        acc_str = "Accuracy trn %6.2f %%, val %6.2f %%" % (100. * trn_acc,
                                                           100. * val_acc)
        print i_str, acc_str, t_str

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

    def calculate_accuracy(self):
        u, y = self.get_data('trn')
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

    def calculate_accuracy(self):
        ut, yt = self.get_data('trn')
        # Training error
        ut, yt = u[:, :self.testset_size], y.T[:self.testset_size].T
        trn_acc = self.accuracy(self.mdl.estimate_y(ut, np.nan * yt, 'training_err'), yt)

        # Validation error
        uv, yv = self.get_data('val')
        val_acc = self.accuracy(self.mdl.estimate_y(uv, np.nan * yv, 'validation'), yv)

        return (trn_acc, val_acc)

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


class SupervisedLearningGUI(SupervisedLearning):
    def run(self):
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
        self.tau = self.tau_start
        self.iter = 0

        self.run_iteration(self.iter, self.mdl, u, y, self.tau)
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
            self.run_iteration(self.iter, mdl, u, y, self.tau)
            self.tau = self.tau * self.tau_alpha + (1 - self.tau_alpha) * self.tau_end
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

    def calculate_accuracy(self):
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
        return super(SupervisedLearningGUI, self).calculate_accuracy()


class MultiModelLearning(UnsupervisedLearning):

    def configure(self):
        layers = [30]
        self.tau_start = 20
        self.tau_end = 5
        self.tau_alpha = 0.99
        self.mdl = []
        for i in range(10):
            self.mdl += [
                ECA(layers, self.data.size('trn', 0)[0][0], 0,
                    #T.tanh)  # T.tanh, rect, None, etc..
                    #lambda x: x)
                    rect)  # T.tanh, rect, None, etc..
            ]

    def run(self):
        print 'Training...'
        u, y = self.get_data('trn')
        tau = self.tau_start
        print u.shape, y.shape
        try:
            for iter in range(self.trn_iters):
                for i in range(10):
                    # Show only samples belonging to a class
                    self.run_iteration(iter, self.mdl[i], u[:, y == i], None, tau)
                if (iter + 50) % 100 == 0:
                    self.print_accuracy(iter)
                tau = self.tau_update(tau)
        except KeyboardInterrupt:
            pass

    def get_data(self, type):
        assert type in ['tst', 'trn', 'val']
        return self.data.get(type, i=0, as_one_hot=False)

    def calculate_accuracy(self):
        # Training error
        u, y = self.get_data('trn')
        r = []
        for i in range(10):
            u_est = self.mdl[i].estimate_u(u, None, 'training_err') * 2
            reconst_mse = np.average(np.square(u_est - u), axis=0)
            r += [reconst_mse]
        y_est = -np.array(r)
        trn_acc = self.accuracy(y_est[:, :self.testset_size],
                                y.T[:self.testset_size].T)

        # Validation error
        u, y = self.get_data('val')
        r = []
        for i in range(10):
            u_est = self.mdl[i].estimate_u(u, None, 'validation') * 2
            reconst_mse = np.average(np.square(u_est - u), axis=0)
            r += [reconst_mse]
        y_est = -np.array(r)
        val_acc = self.accuracy(y_est, y)
        return (trn_acc, val_acc)

def main():
    print 'Initializing...'
    #o = UnsupervisedLearning()
    #o = SupervisedLearningGUI()
    #o = SupervisedLearning()
    o = MultiModelLearning()
    o.run()

if __name__ == '__main__':
    main()
