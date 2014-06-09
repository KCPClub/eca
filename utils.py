import cPickle
import gzip
import os

import theano.tensor as T

import numpy as np

rect = lambda x: T.where(x < 0., 0., x)

def rearrange_for_plot(w):
    """
    Helper for tiling 1-dimensional square vectors into an array of images
    Expecting w in form (n_pixels, n_images)
    """
    assert w.ndim == 2
    w = np.swapaxes(w, 0, 1)
    n_images = w.shape[0]
    ratio = 4/3
    if np.sqrt(w.shape[1]) % 1.0 == 0:
        chans = 1
        image_dim = int(np.sqrt(w.shape[1]))
    elif np.sqrt(w.shape[1] / 3.) % 1.0 == 0:
        chans = 3
        image_dim = int(np.sqrt(w.shape[1] / 3.))
    else:
        print 'Chosen weights probably not representing a square image'
        return w
    l = np.int(np.sqrt(n_images * ratio) + 0.5)
    full_rows = n_images / l
    last_row = n_images % l

    # Scale pixels to interval 0..1
    w += np.abs(np.min(w))
    w /= np.max(w)
    w_ = np.ones((n_images, image_dim + 1, image_dim + 1, chans))
    w_[:, 1:, 1:, :] = w.reshape(n_images, image_dim, image_dim, chans)
    if chans == 1:
        # Remove color channel if this is grayscale image
        w_ = w_.reshape(w_.shape[:-1])

    rows = np.vstack([np.hstack(w_[l * i:l * (i + 1), :, :]) for i in
                      range(full_rows)])
    if last_row:
        r = np.hstack(w_[l * full_rows:, :, :])
        ones = np.ones((r.shape[0], (l - last_row) * (image_dim + 1), chans))
        if chans == 1:
            ones = ones[:, :, 0]
        r = np.hstack((r, ones))
        rows = np.vstack((rows, r))
    return rows


def axis_and_show(axis):
    """ Helper for hiding axis handling """
    if axis is None:
        try:
            import matplotlib.pyplot as axis
        except ImportError:
            pass
        return axis, True, axis.ylim if axis else None
    return axis, False, axis.set_ylim


def imshowtiled(im, axis=None):
    axis, show_it, _ = axis_and_show(axis)
    if axis is None:
        return
    im = rearrange_for_plot(im)
    if im.ndim == 3:
        im = axis.imshow(im, interpolation='nearest')
    elif im.ndim == 2:
        import matplotlib.cm as cm
        im = axis.imshow(im, cmap=cm.Greys_r, interpolation='nearest')
    else:
        pass
    if show_it:
        axis.show()
    return im


def plot_Xdist(signal, axis=None):
    axis, show_it, _ = axis_and_show(axis)
    if axis is None:
        return
    s = signal.val()
    n = s.shape[1] / 10.
    for row in s[:5]:
        p, x = np.histogram(row, bins=n, density=True)
        x = x[:-1] + (x[1] - x[0])/2   # convert bin edges to centers
        axis.plot(x, p)
    if show_it:
        axis.show()


def plot_qXphi(signal, n=int(1e5), axis=None):
    axis, show_it, lim = axis_and_show(axis)
    if axis is None:
        return
    en = np.mean(np.square(signal.val()), axis=1)
    nphi = np.linalg.norm(signal.layer.phi[0].get_value(), axis=0)
    Q = T.diagonal(signal.layer.Q).eval()
    pen, = axis.plot(en[:n], 's-')
    pphi, = axis.plot(nphi[:n], '*-')
    pq, = axis.plot(Q[:n], 'x-')
    axis.legend([pen, pphi, pq], ['E{X^2}', '|phi|', 'q_i'])
    lim([0.0, 5])
    if show_it:
        axis.show()


def plot_svds(*args, **kwargs):
    axis = kwargs['axis'] if 'axis' in kwargs else None
    axis, show_it, _ = axis_and_show(axis)
    if axis is None:
        return
    n = kwargs['n'] if 'n' in kwargs else int(1e9)
    plots = []
    names = []
    svd = lambda x: np.linalg.svd(x, compute_uv=False) / np.sqrt(x.shape[1])
    for s in args:
        try:
            val = s.val()
        except:
            val = s
        plots.append(axis.plot(svd(val)[:n])[-1])
        try:
            name = s.name
        except:
            name = '?'
        names.append('svd(' + name + ')')
    axis.legend(plots, names)
    if show_it:
        axis.show()


def visualize(weights):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        return
    animation = False
    if type(weights) is list and len(weights) > 1:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ims = [[imshowtiled(w, axis=ax)] for w in weight_seq]
        import matplotlib.animation as animation
        ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=1000)
        #writer = animation.writers['ffmpeg'](fps=20,bitrate=1000)
        #ani.save('demo.mp4',writer=writer,dpi=100)
        plt.show()
    else:
        weights = weights[0] if type(weights) is list else weights
        imshowtiled(weights)


class Dataset(object):
    """
    Class for handling training, validation, and test data
    """
    class Data(object):
        def __init__(self, samples, labels, type):
            self.samples = samples
            self.labels = labels
            self.type = type
            self.k = samples.shape[1]
            self.n = samples.shape[0]

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
        self.batch_size = batch_size
        self.testset_size = testset_size
        self.as_one_hot = as_one_hot
        self.stacked = stacked
        assert not stacked or as_one_hot, 'stacking requires one hot'
        self.load()
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
            return Dataset.Data(u, y[start:end].T, type)

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

    def get_patches(self, w=8, m=10000, normalize_contrast=False):
        patches = []
        rng = np.random.RandomState(seed=0)
        pix = self.data['trn'][0]
        pix = pix.reshape((pix.shape[0],) + self.data_shape)
        width, height, chans = self.data_shape
        for i in xrange(m):
            x, y = rng.randint(width - w), rng.randint(height - w)
            j = rng.randint(len(pix))
            patches += [pix[j, x:x+w, y:y+w, :chans].reshape(w * w * chans)]

        patches = np.array(patches)
        if normalize_contrast:
            patches -= np.mean(patches, axis=1, keepdims=True)
            patches /= np.maximum(np.std(patches, axis=1, keepdims=True), 1e-10)

        return patches.T


class MnistDataset(Dataset):
    def load(self):
        # Download e.g. from http://deeplearning.net/data/mnist/mnist.pkl.gz
        filename = 'mnist.pkl.gz'
        if not os.path.exists(filename):
            raise Exception("Dataset not found, please run:\n  wget http://deeplearning.net/data/mnist/mnist.pkl.gz")

        self.data_shape = (28, 28, 1)
        data = cPickle.load(gzip.open(filename))
        self.data = {
            'trn': [np.float32(data[0][0]), np.int32(data[0][1])],
            'val': [np.float32(data[1][0][:self.testset_size]),
                    np.int32(data[1][1][:self.testset_size])],
            'tst': [np.float32(data[2][0][:self.testset_size]),
                    np.int32(data[2][1][:self.testset_size])]
        }


class Cifar10Dataset(Dataset):
    def load(self):
        from skdata.cifar10.dataset import CIFAR10
        c = CIFAR10()
        len(c.meta)
        pix = np.float32(c._pixels / 255.)
        self.data_shape = pix.shape[1:]
        assert self.data_shape == (32, 32, 3)
        pix = pix.reshape(60000, np.prod(self.data_shape))
        lbl = c._labels

        assert self.testset_size <= 10000
        t = self.testset_size
        self.data = {
            'trn': [pix[:40000], lbl[:40000]],
            'val': [pix[40000:40000 + t], lbl[40000:40000 + t]],
            'tst': [pix[50000:50000 + t], lbl[50000:50000 + t]]
        }

def free_mem():
    from theano.sandbox.cuda import cuda_ndarray
    return cuda_ndarray.cuda_ndarray.mem_info()[0] / 1024 / 1024
