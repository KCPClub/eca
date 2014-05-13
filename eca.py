import numpy as np
import time as t
import theano
import theano.tensor as T
DEBUG_INFO = False
FLOATX = theano.config.floatX


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
        self.var = theano.shared(np.float32(rng.uniform(size=(n, k))), name='X')
        #self.value = np.zeros((n, k))
        self.k = k
        self.n = n

    def variance(self):
        return np.average(np.log(np.var(self.var.get_value(), axis=1)))

    def energy(self):
        return np.average(np.square(self.var.get_value()), axis=1)


class Model(object):
    def __init__(self, name, (m, n), nonlin=None, identity=False):
        rng = np.random.RandomState(0)
        self.m = m  # input state size
        self.n = n  # latent space size
        self.model_update_f = {}
        self.state_update_f = {}
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
            E_XU_init = np.identity(self.n, dtype=FLOATX)
        else:
            E_XU_init = np.float32(rng.uniform(size=(n, m)) - 0.5)
        self.E_XU = theano.shared(E_XU_init, name='E_XU')
        self.E_XX = theano.shared(np.identity(n, dtype=FLOATX), name='E_XX')
        self.Q = theano.shared(np.identity(n, dtype=FLOATX), name='Q')
        self.phi = theano.shared(E_XU_init.T, name='phi')
        self.child = None
        self.parent = None
        self.name = name

    def delete_state(self):
        assert len(self.X) > 1, "Cannot delete default state"
        # Discard the last one
        del self.X[-1]
        del self.state_update_f[len(self.X)]
        assert len(self.X) not in self.model_update_f

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
        if self.identity:
            return
        if state_index in self.model_update_f:
            return self.model_update_f[state_index](tau)
        else:
            x = self.X[state_index]
            x_prev = self.parent.X[state_index]
            assert x_prev.k == x.k, "Sample size mismatch"
            assert x_prev.n == self.m, "Input dim mismatch"
            assert x.n == self.n, "Output dim mismatch"
            assert not self.identity, "Trying to update identity layer, probably input"
            k = np.float32(x.k)
            tau_in = T.scalar('tau', dtype=FLOATX)
            (E_XU_new, d1) = lerp(tau_in,
                                  self.E_XU,
                                  T.dot(x.var, x_prev.var.T) / k)
            (E_XX_new, d2) = lerp(tau_in,
                                  self.E_XX,
                                  T.dot(x.var, x.var.T) / k)
            E_XU_update = (self.E_XU, E_XU_new)
            E_XX_update = (self.E_XX, E_XX_new)
            b = 1.
            d = T.diagonal(E_XX_new)
            Q_new = theano.sandbox.linalg.ops.diag(b / T.where(d < 0.05, 0.05, d))
            Q_update = (self.Q, Q_new)

            # TODO: optional spatial neighborhood coupling
            phi_update = (self.phi, T.dot(Q_new, E_XU_new).T)

            self.model_update_f[state_index] = theano.function(
                inputs=[tau_in],
                outputs=[d1, d2],
                updates=[E_XU_update, E_XX_update, Q_update, phi_update])

            # Call self now that we have the function set
            return self.update_model(state_index, tau)

            #self.info('Updating model, avg(diag(E_XX))=%.2f, ' % np.average(d) +
                      #'avg phi col norm %.2f' % np.average(np.sqrt(np.sum(np.square(self.phi),
                                                                           #axis=0))))

    def update_state(self, state_index, input, tau, missing_values=None):
        if state_index in self.state_update_f:
            return self.state_update_f[state_index](tau)
        else:
            tau_in = T.scalar('tau', dtype=FLOATX)
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
                feedforward = T.dot(self.phi.T, self.parent.X[state_index].var)
                origin = self.parent.name
            else:
                origin = 'system input (%d dim)' % input.shape[0]
                self.input = theano.shared(np.float32(input), name='input')
                feedforward = self.input

            #self.info('Updating state[%d]: feedfwd from %s and feedback from %s' %
                      #(state_index, origin, self.child.name if self.child else "nowhere"))


            # Apply nonlinearity to feedforward path only
            if self.nonlin:
                feedforward = self.nonlin(feedforward)

            new_value = feedforward - feedback

            # If predicting missing values, force them to zero in residual so
            # that they don't influence learning
            if missing_values is not None and input is not None:
                self.missing_values = theano.shared(missing_values, name='missings')
                new_value = T.where(missing_values, 0.0, new_value)

            # Or if we want to apply nonlinearity to the result
            #if self.nonlin:
                #new_value = self.nonlin(new_value)

            (new_X, d) = lerp(tau_in, x.var, new_value)

            self.state_update_f[state_index] = theano.function(
                inputs=[tau_in],
                outputs=[d],
                updates=[(x.var, new_X)])

            return self.update_state(state_index, input, tau, missing_values)

    def get_feedback(self, state_index, parent=None):
        return T.dot(self.phi, self.X[state_index].var)

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

    def update_state(self, i, input, tau, missing):
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

    def get_feedback(self, state_index, parent=None):
        assert parent is not None, 'CCA needs parent'
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

    def update(self, u, y, tau, missing=None):
        """
        Performs update round-trip from u to X and back.
        """
        # Create default state now when we know the sample count
        if len(self.l_U.X) == 0:
            for l in self.layers:
                l.create_state(u.shape[1])

        self.update_states(0, u, y, tau, missing)
        self.update_models(0, tau)

    def update_models(self, i, tau):
        for l in self.layers:
            l.update_model(i, tau)

    def update_states(self, i, u, y, tau, missing=None):
        max_diff = -1e100

        # Loop over net inputs
        for (x, val) in zip([self.l_U, self.l_Y], [u, y]):
            # Loop through the chain of layers
            while x:
                d = x.update_state(i, val, tau, missing)
                x = x.child
                val = None  # Only the first layer will have value in

                # For convergence detection
                max_diff = np.max([max_diff, np.average(np.abs(d))])
        return max_diff

    def converge_state(self, i, u, y, tau, missing=None):
        max_delta = 1.0
        iter = 0
        (delta_limit, time_limit, iter_limit) = (1e-3, 20, 100)
        t_start = t.time()
        # Convergence condition, pretty arbitrary for now
        while max_delta > delta_limit and t.time() - t_start < time_limit:
            max_delta = self.update_states(i, u, y, tau, missing)
            iter += 1
            if iter >= iter_limit:
                break
        print 'Converged in', "%.1f" % (t.time() - t_start), 's,', iter,
        print 'iters, delta %.4f' % max_delta,
        print 'Limits: i:', iter_limit, 't:', time_limit, 'd:', delta_limit

    def predict_Y(self, u):
        k = u.shape[1]
        y = np.zeros((self.l_Y.m, k), np.float32) if self.l_Y else None
        # Create a temporary state
        for l in self.layers:
            i = l.create_state(k)
        self.converge_state(i, u, y, 10.)

        # Take a copy, and clean up the temporary state
        val = self.l_Y.child.get_feedback(i).eval()
        for l in self.layers:
            l.delete_state()
        return val

    def converged_U(self, u, missing=None):
        k = u.shape[1]
        y = np.zeros((self.l_Y.m, k), dtype=np.float32) if self.l_Y else None
        # Create a temporary state
        for l in self.layers:
            i = l.create_state(k)
        assert i > 0, '0 is for training data'
        self.converge_state(i, u, y, 5., missing)

        # Take a copy, and clean up the temporary state
        val = self.uest(i)
        for l in self.layers:
            l.delete_state()
        return val

    def converged_X(self, u):
        k = u.shape[1]
        y = np.zeros((self.l_Y.m, k), dtype=np.float32) if self.l_Y else None
        # Create a temporary state
        for l in self.layers:
            i = l.create_state(k)
        assert i > 0, '0 is for training data'
        self.converge_state(i, u, y, 5.)

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
        return l.X[i].var.get_value()

    def uest(self, i=0):
        return self.l_U.child.get_feedback(i).eval()

    def first_phi(self):
        # index is omitted for now, and the lowest layer is plotted
        return self.l_U.child.phi.get_value()

    def variance(self, states=None):
        f = lambda l: (l.name, l.X[0].variance())
        return map(f, self.layers)

    def energy(self):
        f = lambda l: (l.name, np.diagonal(l.E_XX.get_value()))
        return map(f, self.layers)

    def avg_levels(self):
        f = lambda l: (l.name, np.linalg.norm(np.average(l.X[0].var.get_value(), axis=1)))
        return map(f, self.layers)

    def phi_norms(self):
        f = lambda l: (l.name, np.average(np.linalg.norm(l.phi.get_value(), axis=0)))
        return map(f, self.layers)

    def reconstruction_error(self):
        # TODO
        pass
