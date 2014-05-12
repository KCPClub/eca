#!/usr/bin/env python
import numpy as np
import pickle
import gzip
import os
import time
from eca import ECA
import theano.tensor as T

# KNN based algorithm was used to prefill nans in above to form this
DATA_PATH = os.path.join('experiments', 'missing_value')


def prepare_submission(mat, missing):
    # Flatten in matlab order
    res = mat.flatten('F')[missing.flatten('F')]
    index = 0
    print 'Writing submission.csv with', len(res), 'values'
    with open('submission.csv', 'w') as f:
        f.write("Id,Prediction\n")
        for (i, v) in enumerate(missing.flatten('F')):
            if v:
                f.write("%d,%d\n" % (i + 1, res[index]))
                index += 1


class TrainData(object):
    def __init__(self, n_missing, ratio_of_ones, seed):
        """ Creates a new data object with matrix that has values missing and
        marked as nan and correct results.
        """
        self.rng = np.random.RandomState(seed)
        x = pickle.load(gzip.open(os.path.join(DATA_PATH,
                                               'matrix_prefilled.pkl.gz')))
        assert not np.any(np.isnan(x))
        self.original = x
        self.x, self.missing = self.hide_values(self.original, n_missing, ratio_of_ones)

    def hide_values(self, x, n_missing, ratio_of_ones):
        n_ones = int(n_missing * ratio_of_ones)
        mat_ones = np.zeros(shape=x.shape, dtype=bool)
        counter = 0
        while counter < n_ones:
            i = self.rng.randint(x.shape[0])
            j = self.rng.randint(x.shape[1])
            if x[i, j] > 0 and not mat_ones[i, j]:
                mat_ones[i, j] = True
                counter += 1

        n_zeros = n_missing - n_ones
        mat_zeros = np.zeros(shape=x.shape, dtype=bool)
        counter = 0
        while counter < n_zeros:
            i = self.rng.randint(x.shape[0])
            j = self.rng.randint(x.shape[1])
            if x[i, j] == 0 and not mat_zeros[i, j]:
                mat_zeros[i, j] = True
                counter += 1

        assert np.sum(mat_ones + mat_zeros) == n_missing
        missing = mat_ones + mat_zeros

        x = x.copy()
        x[missing] = np.nan
        return (x, missing)

    def accuracy(self, pred):
        p = pred[self.missing]
        r = self.original[self.missing]
        r[r > 0] = 1
        r = r.astype(int)

        assert np.all(p[p != 0] == 1), 'Found non-binary values!!!'
        assert np.all(r[r != 0] == 1), 'Found non-binary values!!!'
        return float(np.sum(p == r)) / len(p)


def assign_60p_to_one(mat, missing):
    """ It is given that 60% of the missing values are ones so this tries to
    adjust predictions so that the largest 60 % of them will be assigned to 1
    """
    v = mat[missing]
    indices = np.argsort(v)
    v[indices[:0.4*len(indices)]] = 0
    v[indices[0.4*len(indices):]] = 1
    mat[missing] = v
    return mat


def eca_missing_value_prediction(data, params):
    if isinstance(data, TrainData):
        x = data.x
        stats = True
    else:
        x = data
        stats = False

    m = ECA(params['layers'], x.T.shape[0], 0, params['nonlin'])

    # Preprocessing, make data binary.
    # Helps with accuracy, not strictly required
    nans = np.isnan(x)
    x[nans] = 0
    x[x > 0] = 1

    (tau, tau_target, alpha) = params['tau']

    acc = np.nan
    best_accuracy = np.nan

    t = time.time()
    # Main loop
    for i in range(params['iters']):
        tau = alpha * tau + (1 - alpha) * tau_target
        m.update(x.T, None, tau, missing=nans.T)

        # Progress reporting
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t
            if stats:
                pred = m.uest().T
                pred = assign_60p_to_one(pred, nans)
                acc = data.accuracy(pred)
                best_accuracy = max(acc, best_accuracy)
            print "A: %5.2f %%" % (acc * 100.),
            print "U var: %.2f" % m.variance()[0][1],
            print "T: %4.1f" % tau,
            print 'U MSE: %.5f' % np.average(np.square(x.T - m.l_U.X[0].var.get_value())),
            print 'Took [s]: %.2f' % elapsed
            t = time.time()

    # Get results
    pred = m.uest().T
    return (assign_60p_to_one(pred, nans), best_accuracy)


def main():
    repeats = 1  # per configuration

    relu = lambda x: T.where(x < 0., 0., x)
    runs = [
        # tanh requires higher tau
        {'config': {'layers': [70], 'tau': (20, 4, 0.95), 'iters': 200, 'nonlin': relu}},
        {'config': {'layers': [60], 'tau': (20, 4, 0.99), 'iters': 200, 'nonlin': T.tanh}},
        {'config': {'layers': [70], 'tau': (20, 4, 0.99), 'iters': 200, 'nonlin': T.tanh}},
    ]

    try:
        for run in runs:
            print 'Running configuration', run['config']
            results = []
            best = 0.0

            # Do several runs with different seeds
            for i in range(repeats):
                d = TrainData(35022, 0.6, i)
                pred, best_iter = eca_missing_value_prediction(d, run['config'])

                # Store results
                acc = d.accuracy(pred)
                results = [acc]
                best = max(best, best_iter)
                print '%5.2f %%' % (acc * 100.), str(run['config']), 'iteration', i

            run['results'] = (results, best)
            print '-----------------'
    except KeyboardInterrupt:
        pass

    print 'Summary'
    print '-------'
    for run in runs:
        if 'results' not in run:
            continue
        res, best = run['results']
        print "%5.2f +- %.2f (best: %5.2f): %s" % (100. * np.mean(res),
                                                   100. * np.sqrt(np.var(res)),
                                                   100. * best,
                                                   str(run['config']),)

    # Prepare a submission file for Kaggle music prediction competition
    # at https://inclass.kaggle.com/c/aalto-music-listening-prediction
    # This submission would've become 5th in the competition. Uncomment:
    #config = {'layers': [70], 'tau': (20, 4, 0.99), 'iters': 200, 'nonlin': np.tanh}
    #x = pickle.load(gzip.open(os.path.join(DATA_PATH, 'matrix_with_nans.pkl')))
    #nans = np.isnan(x)
    #prepare_submission(eca_missing_value_prediction(x, config)[0], nans)

if __name__ == '__main__':
    main()
