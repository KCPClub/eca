#!/usr/bin/env python
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from eca import ECA
from utils import Data

conf = {
    'use_minibatches': False,
    'use_svm': False
}

if conf['use_svm']:
    from sklearn import svm
else:
    from sklearn.linear_model import LogisticRegression


def get_optimal_system_size(input):
    singulars = np.linalg.svd(input)[1]
    for i in range(1, len(singulars)):
        if np.square(singulars[i]) < np.square(np.sum(singulars[:i]) / ((i + 1) * (1 + 1) - 1)):
            print 'Optimal n for linear system would be', i
            break


class TestCaseBase(object):
    def __init__(self):
        self.batch_size = 1000  # Actually training input size (k)
        self.testset_size = 1000
        self.data = Data(batch_size=self.batch_size, testset_size=self.testset_size)
        self.trn_iters = 400
        self.mdl = None
        self.onehot = False
        self.tau_start = None
        self.tau_end = None
        self.tau_alpha = None

        self.configure()
        assert self.mdl is not None

        (u, y) = self.data.get('trn', i=0, as_one_hot=self.onehot)
        #get_optimal_system_size(u)

        # This is global slowness that helps the model to get past the
        # unstable beginning..
        tau = self.tau_start
        if self.onehot:
            y *= 9.2
            print 'energy u:', np.average(np.sum(np.square(u), axis=0)),
            print 'energy y:', np.average(np.sum(np.square(y), axis=0))

        print 'Training...'
        try:
            t = time.time()
            for i in range(self.trn_iters):
                # This does not work. Visualization of FI * X becomes blurred (an
                # average of samples?)
                if conf['use_minibatches']:
                    assert False, "Not supported"
                    u, y = self.data.get('trn', i=0)

                self.mdl.update(u, y, tau)
                tau = tau * self.tau_alpha + (1 - self.tau_alpha) * self.tau_end

                i_str = "I %4d:" % (i + 1)
                # Progress prings
                if ((i + 1) % 20 == 0):
                    t_str = 't: %.2f s' % (time.time() - t)
                    t = time.time()
                    tau_str = "Tau: %4.1f" % tau

                    tostr = lambda t: "{" + ", ".join(["%s: %6.2f" % (n, v) for (n, v) in t]) + "}"

                    var_str = " logvar:" + tostr(self.mdl.variance())
                    a_str   = " avg:   " + tostr(self.mdl.avg_levels())
                    phi_str = " |phi|: " + tostr(self.mdl.phi_norms())

                    print i_str, tau_str, t_str, var_str, phi_str
                    #print var_str, a_str, phi_str
                    #print var_str, phi_str

                # Accuracy prints
                if ((i + 1) % 200 == 0):
                    t2 = time.time()
                    (trn_acc, val_acc) = self.calculate_accuracy(u, y)
                    t_str = 't: %.2f s,' % (time.time() - t2)
                    acc_str = "Accuracy trn %6.2f %%, val %6.2f %%" % (100. * trn_acc,
                                                              100. * val_acc)

                    print i_str, acc_str, t_str

            print 'Testing accuracy...'
            (ut, yt) = self.data.get('tst', as_one_hot=self.onehot)
            # TODO: test error broken at the moment, trust validation as it is not used for
            # hyper-parameter tuning at the moment
            #tst_acc = self.calculate_accuracy(ut, yt)
            tst_acc = np.nan
            print "Accuracy: Training %6.2f %%, validation %6.2f %%, test %6.2f %%" % (
                100. * trn_acc,
                100. * val_acc,
                100. * tst_acc)
        except KeyboardInterrupt:
            pass
        # Viz
        # Plot some energies ("lambdas")
        #e = np.array(self.mdl.energy())
        #f, ax = plt.subplots(len(e))
        #for (i, row) in enumerate(e):
            #ax[i].plot(np.sort(row)[::-1])
            #print "E(X%d)" % i, ", ".join(["%4.2f" % v for v in np.sort(row)[-1:-3:-1]]), '...', \
                #", ".join(["%4.2f" % v for v in np.sort(row)[0:3]])

        #plt.show()
        self.visualize()

    def configure(self):
        raise NotImplemented()

    def accuracy(self, y_est, y_true):
        y_true = y_true if len(y_true.shape) == 1 else np.argmax(y_true, axis=0)
        return float(np.bincount(y_est == y_true, minlength=2)[1]) / len(y_est)

    def calculate_accuracy(self, u, y):
        if conf['use_svm']:
            classifier = svm.LinearSVC()
            print 'Testing accuracy with SVM...'
        else:
            classifier = LogisticRegression()
            print 'Testing accuracy with logistic regression...'

        # y has the corresponding labels to the data used in teaching
        # so we can build the model now
        self.mdl.fit_classifier(classifier, y)

        def estimate(data):
            state = self.mdl.converged_X(data)
            y_est = classifier.predict(state.T)
            return y_est

        y_est = estimate(u[:, :self.testset_size])
        trn_acc = self.accuracy(y_est, y.T[:self.testset_size].T)
        (uv, yv) = self.data.get('val', limit=self.batch_size, as_one_hot=self.onehot)
        val_acc = self.accuracy(estimate(uv), yv)
        return (trn_acc, val_acc)

rect = lambda x: np.where(x < 0., 0., x)

class UnsuperLayer(TestCaseBase):
    def configure(self):
        layers = [70]  # Try e.g. [30, 20] for multiple layers and increase tau start
        self.tau_start = 20
        self.tau_end = 5
        self.tau_alpha = 0.99
        self.mdl = ECA(layers,
                       self.data.size('trn', 0)[0][0],
                       0,  # n of output
                       np.abs) # np.tanh, rect, None, etc..

    def visualize(self):
        plt.imshow(self.mdl.phi_im(1), cmap=cm.Greys_r)
        plt.show()


class SuperLayer(TestCaseBase):
    def configure(self):
        layers = [30]
        self.tau_start = 60
        self.tau_end = 5
        self.tau_alpha = 0.99
        self.onehot = True
        self.mdl = ECA(layers,
                       self.data.size('trn', 0)[0][0],
                       self.data.size('trn', 0, as_one_hot=True)[1][0],
                       rect)

    def visualize(self):
        plt.imshow(self.mdl.phi_im(1), cmap=cm.Greys_r)
        plt.show()

    def calculate_accuracy(self, u, y):
        # Use base class implementation, because CCA regression doesn't work?
        #return super(SuperLayer, self).calculate_accuracy(u, np.argmax(y, axis=0))

        # TODO: This CCA does not give sensible y out, figure out why
        print 'Testing accuracy...'

        def estimate(data, y):
            prediction = self.mdl.predict_Y(data)
            # Print what the two first samples give
            #for i in range(2):
                #print prediction.T[i], np.argmax(prediction.T[i]), np.argmax(y.T[i])
            return np.argmin(prediction, axis=0)

        (ut, yt) = (u[:, :self.batch_size], y.T[:self.batch_size].T)
        y_est = estimate(ut, yt)
        trn_acc = self.accuracy(y_est, yt)
        (uv, yv) = self.data.get('val', limit=self.batch_size, as_one_hot=self.onehot)
        val_acc = self.accuracy(estimate(uv, yv), yv)
        return (trn_acc, val_acc)


def main():
    print 'Initializing...'

    UnsuperLayer()

    # CCA experiment
    #SuperLayer()

    #print 'super 81.68 %%' # Some older results, cannot remember hyper-params
    #print 'one l 88.29 %%'
    #print 'two l 89.42 %%'


if __name__ == '__main__':
    main()
