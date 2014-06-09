from utils import MnistDataset, Cifar10Dataset
import utils
import numpy as np
from eca import ECA, Input, Layer


def test_loading():
    for Dataset in [MnistDataset, Cifar10Dataset]:
        d = Dataset(batch_size=400,
                    testset_size=1000)
        p = d.get_patches(w=8, m=20, normalize_contrast=True)
        assert p.shape[1] == 20
        assert p.shape[0] == 8 * 8 * 3 or p.shape[0] == 8 * 8
        # Ensure all samples really are zero mean and 1 var
        assert np.allclose(np.mean(p, axis=0), 0.0, atol=1e-4)
        assert np.allclose(np.std(p, axis=0), 1.0, atol=1e-4)
        data = d.get('trn')
        assert data.k == 400
        data = d.get('val')
        assert data.k == 400
        data = d.get('tst')
        assert data.k == 400


def test_plots():
    """ Warning, the functionality depends on whether there is matplotlib
    installed on the system. It should not draw anything on the screen. """
    try:
        import matplotlib.pyplot as plt
        _, axis = plt.subplots(1, 1)
    except ImportError:
        axis = None

    # Draw samples tiled
    for Dataset in [MnistDataset, Cifar10Dataset]:
        dataset = Dataset(batch_size=400, testset_size=1000)
        data = dataset.get('trn')
        utils.imshowtiled(data.samples, axis=axis)

    class Model(ECA):
        def structure(self):
            self.U = Input('U', np.prod(data.samples.shape[0]))
            self.X = Layer('X', 3, self.U, utils.rect, 1.0)
            self.X2 = Layer('X2', 3, self.U, None, 1.0)
    mdl = Model()
    sig = mdl.new_signals(data.samples.shape[1])

    # Various plots
    utils.plot_Xdist(sig.U, axis=axis)
    utils.plot_qXphi(sig.U.next, axis=axis)
    utils.plot_svds(data.samples, sig.U, sig.U.next, axis=axis)
