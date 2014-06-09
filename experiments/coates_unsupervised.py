#!/usr/bin/env python
import numpy as np

from eca import ECA, Input, Layer
from utils import imshowtiled, rect, Cifar10Dataset
from sklearn.svm import LinearSVC
import sys


class Unsupervised(object):
    def learn(self):
        pass

    def transform(self, data):
        pass


class ECAUnsupervised(object):
    def __init__(self, patches, dim):
        self.train_data = patches
        self.k = patches.shape[1]
        self.n = n = patches.shape[0]
        self.dim = dim
        self.sig = None

        class Model(ECA):
            def structure(self):
                self.U = Input('U', n)
                self.X = Layer('X1', dim, [self.U], rect,
                               min_tau=0., stiffx=1.0)
        self.mdl = Model()

    def learn(self, iterations):
        trn_sig = self.mdl.new_signals(self.k)
        stiff_start, stiff_end, stiff_decay = (0.5, 0.001, 0.95)
        stiff_update = lambda s: s * stiff_decay + (1 - stiff_decay) * stiff_end
        stiff = stiff_start
        for i in range(1, iterations + 1):
            if i % 10 == 1:
                print 'Iteration', i, '/', iterations + 1
            trn_sig.propagate(self.train_data, None)
            trn_sig.adapt_layers(stiff)
            stiff = stiff_update(stiff)
        return self.mdl.first_phi()

    def transform(self, data):
        k = data.shape[1]
        if self.sig is None or self.sig.k != k:
            self.sig = self.mdl.new_signals(k)
        self.sig.converge(data, None)
        return self.sig.U.next.val()

def run(dry_run=False):
    #47.3 % for M: 10000 K: 100 W: 6 stride: 1 trn_m: 5000 tst_m: 1000
    #60.1 % for M: 10000 K: 1000 W: 6 stride: 1 trn_m: 5000 tst_m: 1000
    #57.4 % for M: 10000 K: 1000 W: 8 stride: 1 trn_m: 5000 tst_m: 1000
    #57.5 % for M: 10000 K: 1000 W: 8 stride: 1 trn_m: 5000 tst_m: 1000
    #58.1 % for M: 20000 K: 1000 W: 8 stride: 1 trn_m: 5000 tst_m: 1000
    #55.1 % for M: 10000 K: 1000 W: 8 stride: 2 trn_m: 5000 tst_m: 1000
    #37.09 % for M: 10000 K: 100 W: 8 stride: 1 trn_m: 50000 tst_m: 10000
    #37.8 % for M: 20000 K: 1000 W: 6 stride: 1 trn_m: 5000 tst_m: 1000
    #59.6 % for M: 10000 K: 1000 W: 6 stride: 1 trn_m: 5000 tst_m: 1000
    #67.36 % for M: 10000 K: 1000 W: 6 stride: 1 trn_m: 50000 tst_m: 10000
    #68.14 % for M: 50000 K: 1000 W: 6 stride: 1 trn_m: 50000 tst_m: 10000 (1000 iters)
    #bug: 66.07 % for M: 50000 K: 1000 W: 6 stride: 1 trn_m: 50000 tst_m: 10000 (buggy)
    #52.0 % for M: 50000 K: 1000 W: 6 stride: 1 trn_m: 5000 tst_m: 1000 (avg)
    #26 was! try again with sum and now with right contrst norm
    if dry_run:
        M = 2
        K = 2
        W = 4
        stride = 1
        train_size = 10
        test_size = 1
        train_iters = 1
        normalize = True
    else:
        M = 20000 if len(sys.argv) <= 1 else int(sys.argv[1])
        K = 1000 if len(sys.argv) <= 2 else int(sys.argv[2])
        W = 8 if len(sys.argv) <= 3 else int(sys.argv[3])
        stride = 1 if len(sys.argv) <= 4 else int(sys.argv[4])
        train_size = 5000 if len(sys.argv) <= 5 else int(sys.argv[5])
        test_size = 1000 if len(sys.argv) <= 6 else int(sys.argv[6])
        train_iters = 500 if len(sys.argv) <= 7 else int(sys.argv[7])
        normalize = True if len(sys.argv) <= 8 else bool(int(sys.argv[7]))


    print 'Loading dataset...'
    data = Cifar10Dataset(batch_size=train_size,
                          testset_size=test_size,
                          normalize=False)
    try:
        print 'Preparing patches...'
        patches = data.get_patches(W, M, normalize_contrast=normalize)
        unsup = ECAUnsupervised(patches, K)
        print 'Pretraining with only', train_iters
        feats = unsup.learn(train_iters)
        if not dry_run:
            imshowtiled(feats)

        print 'Applying convolution to train data'

        def convolve(inp, s):
            samples = inp.shape[1]
            conv = np.zeros((2, 2, K, samples))
            assert inp.shape == (32*32*3, samples)
            d = inp.reshape(32, 32, 3, samples)
            for x in xrange(0, 32 - W + 1, s):
                print 'Row', x // s + 1
                for y in xrange(0, 32 - W + 1, s):
                    latent = unsup.transform(d[x:x+W, y:y+W, :, :].reshape(W * W * 3, samples))
                    conv[x // 16, y // 16, :, :] += latent
                    #conv[x // 16, y // 16, :, :] = np.maximum(conv[x // 16, y // 16, :, :], latent)
            return conv.reshape(4 * K, samples)

        # Fitting SVN with input data
        print 'Extracting features for training data'
        d = data.get('trn')
        trn_feats = convolve(d.samples, stride)
        print 'Training SVM'
        clf = LinearSVC()
        clf.fit(trn_feats.T, d.labels)

        # Fitting SVN with input data
        print 'Extracting features for validation data'
        d = data.get('val')
        val_feats = convolve(d.samples, stride)
        print 'Predicting results'
        pred = clf.predict(val_feats.T)
        print 100.0 * d.accuracy(pred), '% for',
        print 'M:', M,
        print 'K:', K,
        print 'W:', W,
        print 'stride:', stride,
        print 'trn_m:', train_size,
        print 'tst_m:', test_size,
        print

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    run()
