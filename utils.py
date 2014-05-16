import cPickle
import gzip
import os

import numpy as np


def rearrange_for_plot(w):
    """
    Helper for tiling 1-dimensional square vectors into an array of images
    """
    ratio = 2/1
    image_dim = np.sqrt(w.shape[0])
    if image_dim - np.floor(image_dim) > 0.001:
        print 'Chosen weights probably not representing a square image'
        return w
    image_dim = int(image_dim)
    l = np.int(np.sqrt(w.shape[1] * ratio)) + 1
    h = np.max([l / ratio, 2])
    #print l, h
    w_ = w.T.reshape(w.shape[1], image_dim, image_dim)
    # TODO: A mess. Fix and clean up. Currently loses plots of some neurons!

    rows = np.vstack([np.hstack(w_[l * i:l * (i + 1), :, :]) for i in
                      range(h-1)])
    #print np.array(rows.shape) / 28
    if l * (h) < w_.shape[0] - 1:
        last_row = np.hstack(w_[l * (h - 1):, :, :])
        last_row = np.hstack((last_row, np.zeros((last_row.shape[0],
                                                  rows.shape[1] -
                                                  last_row.shape[1]))))
        rows = np.vstack((rows, last_row))
    return rows


class MnistData(object):
    """
    Class for handling training, validation, and test data
    """
    def __init__(self, batch_size=500, testset_size=10000, normalize=True):
        # Download e.g. from http://deeplearning.net/data/mnist/mnist.pkl.gz
        filename = 'mnist.pkl.gz'
        if not os.path.exists(filename):
            raise Exception("Dataset not found, please run:\n  wget http://deeplearning.net/data/mnist/mnist.pkl.gz")

        data = cPickle.load(gzip.open(filename))
        self.batch_size = batch_size
        self.testset_size = testset_size
        self.data = {
            'trn': [np.float32(data[0][0]), np.int32(data[0][1])],
            'val': [np.float32(data[1][0][:testset_size]),
                    np.int32(data[1][1][:testset_size])],
            'tst': [np.float32(data[2][0][:testset_size]),
                    np.int32(data[2][1][:testset_size])]
        }
        if normalize:
            for x, y in self.data.values():
                x -= np.mean(x, axis=0, keepdims=True)
                x /= np.maximum(np.std(x, axis=0, keepdims=True), 1e-10)

    def size(self, type, i=None, as_one_hot=False):
        assert type in self.data.keys(), 'type has to be in %s' % str(self.data.keys())
        y_dim = 10 if as_one_hot else 1
        batch_size = self.data[type][0].shape[0] if i is None else self.batch_size
        return ((self.data[type][0].shape[1], batch_size), (y_dim, batch_size))

    def get(self, type, i=None, as_one_hot=False):
        """
        Returns a tuple (u, y) of i'th minibatch expanded into a one-hot coded vectors if necessary.

        E.g. 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        """
        assert type in self.data.keys(), 'type has to be in %s' % str(self.data.keys())
        (u, y) = self.data[type]

        i = 0 if i is None else i
        start = i * self.batch_size
        end = min(u.shape[0], (i + 1) * self.batch_size)

        u = u[start:end].T

        if not as_one_hot:
            return (u, y[start:end].T)

        # Convert into one_hot presentation 2 -> [0, 0, 1, 0, ...]
        y_ = np.array(np.zeros(10))
        y_[y[start]] = 1.
        for i in range(start + 1, end):
            new = np.zeros(10)
            new[y[i]] = 1.
            y_ = np.vstack((y_, new))
        y_ = np.float32(y_)
        return (u, y_.T)


