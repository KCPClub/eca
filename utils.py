import cPickle
import gzip
import os

import theano.tensor as T

import numpy as np

rect = lambda x: T.where(x < 0., 0., x)

def rearrange_for_plot(w):
    """
    Helper for tiling 1-dimensional square vectors into an array of images
    """
    ratio = 4/3
    image_dim = np.sqrt(w.shape[0])
    if image_dim - np.floor(image_dim) > 0.001:
        print 'Chosen weights probably not representing a square image'
        return w
    image_dim = int(image_dim)
    n_images = w.shape[1]
    l = np.int(np.sqrt(n_images * ratio) + 0.5)
    full_rows = n_images / l
    last_row = n_images % l

    w_ = w.T.reshape(n_images, image_dim, image_dim)

    rows = np.vstack([np.hstack(w_[l * i:l * (i + 1), :, :]) for i in
                      range(full_rows)])
    if last_row:
        r = np.hstack(w_[l * full_rows:, :, :])
        r = np.hstack((r, np.zeros((r.shape[0], (l - last_row) * image_dim))))
        rows = np.vstack((rows, r))
    return rows


def visualize(weights):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    animation = False
    if type(weights) is list and len(weights) > 1:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ims = [[ax.imshow(rearrange_for_plot(w), cmap=cm.Greys_r)] for w in weight_seq]
        import matplotlib.animation as animation
        ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=1000)
        #writer = animation.writers['ffmpeg'](fps=20,bitrate=1000)
        #ani.save('demo.mp4',writer=writer,dpi=100)
        plt.show()
    else:
        weights = weights[0] if type(weights) is list else weights
        plt.imshow(rearrange_for_plot(weights), cmap=cm.Greys_r)
        plt.show()


class MnistDataset(object):
    """
    Class for handling training, validation, and test data
    """
    class Data(object):
        def __init__(self, samples, labels, type):
            self.samples = samples
            self.labels = labels
            self.type = type

        def accuracy(self, est, print_it=False):
            # If estimate is stacked, extract the labels
            if est.shape[0] == self.samples.shape[0]:
                est = est[-self.labels.shape[0]:, :]
            true = self.labels if self.labels.ndim == 1 else np.argmax(self.labels, axis=0)
            est = est if est.ndim == 1 else np.argmax(est, axis=0)
            acc = float(np.bincount(est == true, minlength=2)[1]) / len(est)
            if print_it:
                print "Accuracy %s: %6.2f %%" % (self.type, 100. * acc)
            return acc

    def __init__(self, batch_size=500, testset_size=10000, normalize=True,
                 as_one_hot=False, stacked=False):
        # Download e.g. from http://deeplearning.net/data/mnist/mnist.pkl.gz
        filename = 'mnist.pkl.gz'
        if not os.path.exists(filename):
            raise Exception("Dataset not found, please run:\n  wget http://deeplearning.net/data/mnist/mnist.pkl.gz")

        data = cPickle.load(gzip.open(filename))
        self.batch_size = batch_size
        self.testset_size = testset_size
        self.as_one_hot = as_one_hot
        self.stacked = stacked
        assert not stacked or as_one_hot, 'stacking requires one hot'
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

    def size(self, type):
        assert type in self.data.keys(), 'type has to be in %s' % str(self.data.keys())
        u_dim, y_dim = self.dims(type)
        samples = self.samples(type)
        return ((u_dim, samples), (y_dim, samples))

    def samples(self, type):
        return self.data[type][0][:self.batch_size].shape[0]

    def dims(self, type):
        y_dim = 10 if self.as_one_hot else 1
        u_dim = self.data[type][0].shape[1]
        u_dim += y_dim if self.stacked else 0
        return (u_dim, y_dim)

    def get(self, type, i=None):
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

        if not self.as_one_hot:
            return MnistDataset.Data(u, y[start:end].T, type)

        # Convert into one_hot presentation 2 -> [0, 0, 1, 0, ...]
        y_ = np.array(np.zeros(10))
        y_[y[start]] = 1.
        for i in range(start + 1, end):
            new = np.zeros(10)
            new[y[i]] = 1.
            y_ = np.vstack((y_, new))
        y_ = np.float32(y_.T)
        if self.stacked:
            if type == 'trn':
                u = np.vstack([u, y_])
            else:
                u = np.vstack([u, np.float32(np.nan * y_)])
        return MnistDataset.Data(u, y_, type)


def free_mem():
    from theano.sandbox.cuda import cuda_ndarray
    return cuda_ndarray.cuda_ndarray.mem_info()[0] / 1024 / 1024

