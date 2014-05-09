import numpy as np
import time as t
from utils import rearrange_for_plot
DEBUG_INFO = False


def lerp(t, old, new):
    """
    Return new interpolated value and a relative difference
    """
    return ((1 - 1 / t) * old + 1 / t * new,
            (old - new) / (old + 1e-10))


class State(object):
    """ Object that represents any kind of state U, X, X_y, z, ...
    """
    def __init__(self, n, k):
        rng = np.random.RandomState(0)
        self.value = rng.uniform(size=(n, k))
        #self.value = np.zeros((n, k))
        self.k = k
        self.n = n

    def variance(self):
        return np.average(np.log(np.var(self.value, axis=1)))

    def energy(self):
        return np.average(np.square(self.value), axis=1)


class Model(object):
    def __init__(self, name, (m, n), nonlin=None, identity=False):
        rng = np.random.RandomState(0)
        self.m = m  # input state size
        self.n = n  # latent space size
        self.nonlin = nonlin
        self.identity = identity
        # This list holds all the fast state information the model has.
        # Because the model can be used with data of various lengths k, and
        # data sets, different data instances are put here so that 0
        # corresponds to the training data. Indices 1- are available for the
        # model user
        self.X = []
        if identity:
            assert self.m == self.n, "Identity phi requires matching dimensions"
            self.E_XU = np.identity(self.n)
        else:
            self.E_XU = rng.uniform(size=(n, m)) - 0.5
        self.E_XX = np.identity(n)
        self.Q = np.identity(n)
        self.phi = np.dot(self.Q, self.E_XU).T
        self.child = None
        self.parent = None
        self.name = name

    def delete_state(self):
        assert len(self.X) > 1, "Cannot delete default state"
        # Discard the last one
        self.X = self.X[:-1]

    def create_state(self, k):
        self.X.append(State(self.n, k))
        # Return the index of the created state
        return len(self.X) - 1

    def connect_parent(self, parent):
        self.info('has parent %s' % parent.name)
        assert self.m == parent.n, "Child dimensionality doesn't match"
        assert self.child is None
        assert self.parent is None
        self.parent = parent
        parent.child = self

    def update_model(self, state_index, tau):
        # Input layer has identity phi, so no action
        if self.identity:
            return

        x = self.X[state_index]
        x_prev = self.parent.X[state_index]
        assert x_prev.k == x.k, "Sample size mismatch"
        assert x_prev.n == self.m, "Input dim mismatch"
        assert x.n == self.n, "Output dim mismatch"
        assert not self.identity, "Trying to update identity layer, probably input"
        k = x.k
        (self.E_XU, d1) = lerp(tau,
                               self.E_XU,
                               np.dot(x.value, x_prev.value.T) / k)

        (self.E_XX, d2) = lerp(tau,
                               self.E_XX,
                               np.dot(x.value, x.value.T) / k)
        b = 1.
        d = np.diagonal(self.E_XX)
        np.fill_diagonal(self.Q, b / np.where(d < 0.05, 0.05, d))

        # TODO: optional spatial neighborhood coupling
        self.phi = np.dot(self.Q, self.E_XU).T

        self.info('Updating model, avg(diag(E_XX))=%.2f, ' % np.average(d) +
                  'avg phi col norm %.2f' % np.average(np.sqrt(np.sum(np.square(self.phi),
                                                                       axis=0))))

        return (d1, d2)

    def update_state(self, state_index, input, tau):
        x = self.X[state_index]
        assert input is None or x.k == input.shape[1], "Sample size mismatch"

        # If there is a layer following thise, get feedback
        if self.child:
            feedback = self.child.get_feedback(state_index, self)
        else:
            feedback = 0.

        # Feedforward originates from previous layer's state or given input
        if input is None:
            assert not self.identity
            input = self.parent.X[state_index].value
            origin = self.parent.name
        else:
            origin = 'system input (%d dim)' % input.shape[0]

        #self.info('Updating state[%d]: feedfwd from %s and feedback from %s' %
                  #(state_index, origin, self.child.name if self.child else "nowhere"))

        feedforward = np.dot(self.phi.T, input)

        # Apply nonlinearity to feedforward path only
        if self.nonlin:
            feedforward = self.nonlin(feedforward)

        new_value = feedforward - feedback

        # Or if we want to apply nonlinearity to the result
        #if self.nonlin:
            #new_value = self.nonlin(new_value)

        (x.value, d) = lerp(tau, x.value, new_value)
        return d

    def get_feedback(self, state_index, parent):
        return np.dot(self.phi, self.X[state_index].value)

    def info(self, str):
        if DEBUG_INFO:
            print '%5s:' % self.name, str


class CCAModel(Model):
    def __init__(self, name, (m, n)):
        self.E_ZZ = []
        super(CCAModel, self).__init__(name, (m, n), nonlin=None,
                                       identity=False)
        # Phi of this layer is not in use, mark it as nan
        self.phi = np.zeros((1, 1))

    def update_model(self, state_index, tau):
        """ There is no global model state for CCA layer, instead, everything
        happens in update_state
        """
        return

    def create_state(self, k):
        # Create a stack of E_ZZ in addition to base class implementation
        self.E_ZZ.append(State(1, 1))
        return super(CCAModel, self).create_state(k)

    def delete_state(self):
        # Discard the last one
        self.E_ZZ = self.E_ZZ[:-1]
        return super(CCAModel, self).delete_state()

    def update_state(self, i, input, tau):
        z = self.X[i]
        E_ZZ = self.E_ZZ[i]
        assert input is None, "CCA state cannot use input"
        assert self.child is None, 'CCA cannot have children'
        assert len(self.parent) == 2, 'CCA should have exactly 2 parents'
        assert z.n == self.n, "Output dim mismatch"

        # Update state-specific E_ZZ and calculate q for z update
        (E_ZZ.value, di) = lerp(tau * 3.,
                                E_ZZ.value,
                                np.dot(z.value, z.value.T) / z.k)
        assert E_ZZ.value.shape == (1, 1), 'E_ZZ is not a scalar!?'
        b = 1.
        q = b / np.max([0.05, E_ZZ.value[0, 0]])

        # Update z
        [x1, x2] = [self.parent[j].X[i].value for j in (0, 1)]
        new_value = q * np.sum(x1 * x2, axis=0)

        (z.value, dz) = lerp(tau, z.value, new_value)

        self.E_XU = None
        return np.max((np.abs(di), np.average(np.abs(dz))))

    def get_feedback(self, state_index, parent):
        assert len(self.parent) == 2, 'CCA should have exactly 2 parents'
        # XXX: Uncomment the following to avoid signal propagation
        # back from CCA layer
        #return np.zeros(())

        z = self.X[state_index]
        # Find the other branch connecting to this z
        x_other = list(set(self.parent) - set([parent]))[0]

        # This is roughly: x_u_delta = self.Q * x_y * z
        phi = self.Q * x_other.X[state_index].value
        return phi * z.value

    def connect_parent(self, parent):
        self.info('has parent %s' % parent.name)
        assert self.m == parent.n, "Child dimensionality doesn't match"
        assert self.child is None
        self.parent = parent if self.parent is None else [self.parent, parent]
        parent.child = self


class ECA(object):
    """
    Constructs chains of layers connected together. Each layer contains phi
    matrix and corresponding state vector X.

    Typical loop u -- U -- X would represented by two of these layers; one for
    U and another one for X.

    U branch would look something like this, where --- indicates state
    and /\\/ mapping:

      ------------  u

      //|\//\\/\|/  phi_0 = identity
     -------------- layer x0 = U = phi_1^T x-1 - phi_1 X1 = u - phi_1 X1

      //|\//\\/\|/  phi_1 (learned mapping)
     -------------- layer x1 = phi_1^T x0 - phi_2 x2

      //|\//\\/\|/  phi_2 (learned mapping)
     -------------- layer x2 = phi_2^T x1 - phi_3 x3
          .
          .
          .
    """

    def __init__(self, n_layers, n_u, n_y, nonlin=None):
        # 1. Create layers for u branch
        n_all = [n_u] + n_layers
        print 'Creating', len(n_all), 'layers with n in', n_all, 'including input layer U'

        # First the input layer U
        self.l_U = Model('U', (n_u, n_u), None, True)
        self.layers = [self.l_U]

        # Then the consecutive layers U -> X_u1 -> X_u2 -> ...
        for i in range(len(n_all) - 1):
            name = "X_u%d" % (i + 1)
            self.layers.append(Model(name, (n_all[i], n_all[i + 1]), nonlin))
            self.layers[-1].connect_parent(self.layers[-2])

        last_of_u_branch = self.layers[-1]
        self.l_Y = None

        # No supervised y signal => all done!
        if n_y is 0:
            return

        # 2. Create layers for y branch if still here
        # Keep y branch shorter by only taking the last layer dimension
        n_all = [n_y] + [n_layers[-1]]
        print 'Creating', len(n_all), 'layers with n in', n_all, 'including input layer Y'

        # First the input layer Y
        self.l_Y = Model('Y', (n_y, n_y), None, True)
        self.layers += [self.l_Y]

        # Then the consecutive layers Y -> X_y1 -> X_y2 -> ...
        for i in range(len(n_all) - 1):
            name = "X_y%d" % (i + 1)
            self.layers.append(Model(name, (n_all[i], n_all[i + 1]), nonlin))
            self.layers[-1].connect_parent(self.layers[-2])

        # Finally add CCA layer to connect u and y branches:
        # ... -> X_uN -> z
        # ... -> X_yN --'
        print 'Creating CCA layer %d -> %d' % (n_all[-1], 1),
        print 'with parents:', self.layers[-2].name, last_of_u_branch.name
        self.cca = CCAModel('z', (n_all[-1], 1))
        self.layers += [self.cca]
        self.cca.connect_parent(self.layers[-2])
        self.cca.connect_parent(last_of_u_branch)

    def update(self, u, y, tau):
        """
        Performs update round-trip from u to X and back.
        """
        # Create default state now when we know the sample count
        if len(self.l_U.X) == 0:
            for l in self.layers:
                l.create_state(u.shape[1])

        self.update_states(0, u, y, tau)
        self.update_models(0, tau)

    def update_models(self, i, tau):
        for l in self.layers:
            l.update_model(i, tau)

    def update_states(self, i, u, y, tau):
        max_diff = -1e100

        # Loop over net inputs
        for (x, val) in zip([self.l_U, self.l_Y], [u, y]):
            # Loop through the chain of layers
            while x:
                d = x.update_state(i, val, tau)
                x = x.child
                val = None  # Only the first layer will have value in

                # For convergence detection
                max_diff = np.max([max_diff, np.average(np.abs(d))])
        return max_diff

    def converge_state(self, i, u, y, tau):
        max_delta = 1.0
        iter = 0
        (delta_limit, time_limit, iter_limit) = (1e-3, 20, 100)
        t_start = t.time()
        # Convergence condition, pretty arbitrary for now
        while max_delta > delta_limit and t.time() - t_start < time_limit:
            max_delta = self.update_states(i, u, y, tau)
            iter += 1
            if iter >= iter_limit:
                break
        print 'Converged in', "%.1f" % (t.time() - t_start), 's,', iter,
        print 'iters, delta %.4f' % max_delta,
        print 'Limits: i:', iter_limit, 't:', time_limit, 'd:', delta_limit

    def predict_Y(self, u):
        k = u.shape[1]
        y = np.zeros((self.l_Y.m, k)) if self.l_Y else None
        # Create a temporary state
        for l in self.layers:
            i = l.create_state(k)
        self.converge_state(i, u, y, tau=10.)

        # Take a copy, and clean up the temporary state
        val = self.l_Y.X[i].value.copy()
        for l in self.layers:
            l.delete_state()
        return val

    def converged_X(self, u):
        k = u.shape[1]
        y = np.zeros((self.l_Y.m, k)) if self.l_Y else None
        # Create a temporary state
        for l in self.layers:
            i = l.create_state(k)
        assert i > 0, '0 is for training data'
        self.converge_state(i, u, y, tau=5.)

        # Take a copy, and clean up the temporary state
        val = self.latent_layer(i).copy()
        for l in self.layers:
            l.delete_state()
        return val

    def fit_classifier(self, classifier, y):
        classifier.fit(self.latent_layer(0).T, y)

    def latent_layer(self, i):
        """
        Finds the latent representation for classification. The last or one
        before CCA.
        """
        l = self.l_U
        while l.child and not isinstance(l.child, CCAModel):
            l = l.child
        return l.X[i].value

    def phi_im(self, index):
        # index is omitted for now, and the lowest layer is plotted
        return rearrange_for_plot(self.l_U.child.phi)

    def variance(self, states=None):
        f = lambda l: (l.name, l.X[0].variance())
        return map(f, self.layers)

    def energy(self):
        f = lambda l: (l.name, np.diagonal(l.E_XX))
        return map(f, self.layers)

    def avg_levels(self):
        f = lambda l: (l.name, np.linalg.norm(np.average(l.X[0].value, axis=1)))
        return map(f, self.layers)

    def phi_norms(self):
        f = lambda l: (l.name, np.average(np.linalg.norm(l.phi, axis=0)))
        return map(f, self.layers)

    def reconstruction_error(self):
        # TODO
        pass
