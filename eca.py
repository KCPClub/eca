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
        self.k = k
        self.n = n

    def variance(self):
        return np.average(np.log(np.var(self.var.get_value(), axis=1)))

    def energy(self):
        return np.average(np.square(self.var.get_value()), axis=1)


class Model(object):
    def __init__(self, name, n, parent, nonlin, identity=False):
        m = n if parent is None else parent.n
        rng = np.random.RandomState(0)
        self.n = n
        self.m = m
        self.model_update_f = {}
        self.state_update_f = {}
        self.nonlin = nonlin
        # Nonlinearity applied to the estimate coming from child
        self.nonlin_est = lambda x: x
        # This list holds all the fast state information the model has.
        # Because the model can be used with data of various lengths k, and
        # data sets, different data instances are put here so that 0
        # corresponds to the training data. Indices 1- are available for the
        # model user
        self.X = []
        rand_init = np.float32(rng.uniform(size=(n, m)) - 0.5)
        self.E_XU = theano.shared(rand_init, name='E_XU')
        self.E_XX = theano.shared(np.identity(n, dtype=FLOATX), name='E_XX')
        self.Q = theano.shared(np.identity(n, dtype=FLOATX), name='Q')
        self.phi = theano.shared(rand_init.T, name='phi')
        self.child = None
        self.name = name
        self.missing_values = None
        self.parent = parent
        self.custom_state_op = None

        if parent:
            assert parent.child is None
            parent.child = self

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

    def update_model(self, i, tau):
        if i in self.model_update_f:
            return self.model_update_f[i](tau)
        else:
            x = self.X[i]
            x_prev = self.parent.X[i]
            assert x_prev.k == x.k, "Sample size mismatch"
            assert x_prev.n == self.m, "Input dim mismatch"
            assert x.n == self.n, "Output dim mismatch"
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
            Q_new = theano.sandbox.linalg.ops.diag(b / T.where(d < 0.0005, 0.0005, d))
            Q_update = (self.Q, Q_new)

            # TODO: optional spatial neighborhood coupling
            phi_update = (self.phi, T.dot(Q_new, E_XU_new).T)

            self.model_update_f[i] = theano.function(
                inputs=[tau_in],
                outputs=[d1, d2],
                updates=[E_XU_update, E_XX_update, Q_update, phi_update])

            self.info('Model update between ' + self.name + ' and ' + self.parent.name)

            #self.info('Updating model, avg(diag(E_XX))=%.2f, ' % np.average(d) +
                      #'avg phi col norm %.2f' % np.average(np.sqrt(np.sum(np.square(self.phi),
                                                                           #axis=0))))
            # Call self now that we have the function set
            return self.update_model(i, tau)


    def update_state(self, i, input, tau):
        if i in self.state_update_f:
            return self.state_update_f[i](tau)
        else:
            tau_in = T.scalar('tau', dtype=FLOATX)
            x = self.X[i]
            assert input is None or x.k == input.shape[1], "Sample size mismatch"

            # Get estimate of the state from layer above
            estimate = self.estimate(i)

            # Feedforward originates from previous layer's state or given input
            if input is None:
                feedforward = self.feedforward(i)
            else:
                self.missing_values = theano.shared(np.float32(~np.isnan(input)), name='missings')
                input[np.isnan(input)] = 0.0
                self.input = theano.shared(np.float32(input), name='input')
                feedforward = self.input

            self.info('Updating state[%d]: feedfwd from %s and estimate from %s' %
                      (i, self.parent.name if self.parent else 'input',
                          self.child.name if self.child else "nowhere"))

            # Apply nonlinearity to feedforward path only
            if self.nonlin:
                feedforward = self.nonlin(feedforward)

            if self.custom_state_op:
                new_value = self.custom_state_op(feedforward, estimate)
            else:
                new_value = feedforward - estimate

            # If predicting missing values, force them to zero in residual so
            # that they don't influence learning
            if self.missing_values is not None:
                new_value = new_value * self.missing_values

            (new_X, d) = lerp(tau_in, x.var, new_value)

            self.state_update_f[i] = theano.function(
                inputs=[tau_in],
                outputs=[d],
                updates=[(x.var, new_X)])

            return self.update_state(i, input, tau)

    def estimate(self, i):
        """ Ask the child for feedback and apply nonlinearity """
        if not self.child:
            return 0.0
        fb = self.child.feedback(i, self)
        return self.nonlin_est(fb)

    def feedback(self, i, parent=None):
        return T.dot(self.phi, self.X[i].var)

    def feedforward(self, i):
        return T.dot(self.phi.T, self.parent.X[i].var)

    def info(self, str):
        if DEBUG_INFO:
            print '%5s:' % self.name, str


class InputModel(Model):
    def __init__(self, name, n):
        super(InputModel, self).__init__(name,  n, None, None, True)

    def update_model(self, i, tau):
        pass


class CollisionModel(Model):
    def __init__(self, name, n, (parent1, parent2), nonlin=None):
        relu = lambda x: T.where(x < 0, 0, x)
        self.u_side = Model(name + '1', n, parent1, lambda x: nonlin(x))
        self.y_side = Model(name + '2', n, parent2, lambda x: nonlin(x))

        # To make u and y update the same shared state, it must happen
        # simultaneously, so route u to ask y for its feedback as an estimate.
        self.u_side.estimate = lambda i: nonlin(self.y_side.feedforward(i))
        self.u_side.custom_state_op = lambda fromu, fromy: (fromu + fromy)/2.
        #self.u_side.custom_state_op = lambda fromu, fromy: fromu
        #self.u_side.custom_state_op = lambda fromu, fromy: T.sqrt(fromu * fromy + 0.001)

        # Disable state updates on the y side so that X is updated only once
        self.y_side.update_state = lambda i, input, tau: 0.0

        self.n = n
        self.X = []
        self.name = name
        self.phi = self.u_side.phi  # For visualization

    def update_model(self, i, tau):
        self.u_side.update_model(i, tau)
        self.y_side.update_model(i, tau)
        # See what is contribution of u and y to the X
        #print 'y sum', np.average(self.u_side.estimate(i).eval())
        #print 'u sum', np.average(self.u_side.nonlin(self.u_side.feedforward(i)).eval())

        # TODO: Handle model update deltas
        return (0, 0)

    def update_state(self, i, input, tau):
        assert False, 'should be never called'

    def delete_state(self):
        assert len(self.X) > 1, "Cannot delete default state"
        del self.X[-1]
        del self.u_side.state_update_f[len(self.X)]
        assert self.y_side.state_update_f is None

    def create_state(self, k):
        self.X.append(State(self.n, k))
        self.u_side.X = self.X
        self.y_side.X = self.X
        return len(self.X) - 1

    def get_feedback(self, i, parent=None):
        assert False, "should not be called"


class CCAModel(Model):
    def __init__(self, name, (m, n)):
        self.E_ZZ = []
        super(CCAModel, self).__init__(name, (m, n), nonlin=None,
                                       identity=False)
        # Phi of this layer is not in use, mark it as nan
        self.phi = np.zeros((1, 1))

    def update_model(self, i, tau):
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

    def get_feedback(self, i, parent=None):
        assert parent is not None, 'CCA needs parent'
        assert len(self.parent) == 2, 'CCA should have exactly 2 parents'
        # XXX: Uncomment the following to avoid signal propagation
        # back from CCA layer
        #return np.zeros(())

        z = self.X[i]
        # Find the other branch connecting to this z
        x_other = list(set(self.parent) - set([parent]))[0]

        # This is roughly: x_u_delta = self.Q * x_y * z
        phi = self.Q * x_other.X[i].value
        return phi * z.value


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
        print 'Creating', len(n_layers), 'layers with n in', n_layers

        # First the input layer U
        self.l_U = InputModel('U', n_u)
        self.layers = layers = [self.l_U]
        #self.l_U.nonlin_est = lambda x: T.eq(x, T.max(x, axis=0, keepdims=True))
        #self.l_U.nonlin_est = lambda x : T.nnet.sigmoid(x)
        #self.l_U.nonlin_est = nonlin

        # Then the consecutive layers U -> X_u1 -> X_u2 -> ...
        n_ulayers = n_layers[:-1] if n_y else n_layers
        for (i, n) in enumerate(n_ulayers):
            m = Model("X_u%d" % (i + 1), n, layers[-1], nonlin)
            layers.append(m)

        self.l_Y = None

        # No supervised y signal => all done!
        if not n_y:
            return

        # 2. Create layers for y branch if still here
        # Keep y branch shorter by only taking the last layer dimension
        print 'Creating layer (%d) for classification' % n_y

        # First the input layer Y
        self.l_Y = InputModel('Y', n_y)
        layers += [self.l_Y]
        #self.l_Y.nonlin_est = lambda x: T.eq(x, T.max(x, axis=0, keepdims=True))
        #self.l_Y.nonlin_est = lambda x : T.nnet.sigmoid(x)

        # Then add connecting layer to connect u and y branches:
        # ... -> X_uN -> Z
        # ... -> Y    --'
        print 'Creating collision layer with parents:', layers[-2].name, layers[-1].name
        cl = CollisionModel('Z', n_layers[-1], (layers[-2], layers[-1]), nonlin)
        layers += [cl]

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
        max_diff = 1.0

        # Loop over net inputs
        for (x, val) in zip([self.l_U, self.l_Y], [u, y]):
            # Loop through the chain of layers
            while x:
                d = x.update_state(i, val, tau)
                x = x.child
                val = None  # Only the first layer will have value in

                # For convergence detection
                # Disabled because it was taking too much time on CPU
                # Move to GPU!
                #max_diff = np.max([max_diff, np.average(np.abs(d))])
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
        if False:
            print 'Converged in', "%.1f" % (t.time() - t_start), 's,', iter,
            print 'iters, delta %.4f' % max_delta,
            print 'Limits: i:', iter_limit, 't:', time_limit, 'd:', delta_limit

    def converge(self, u, y, f):
        k = u.shape[1]
        # Create a temporary state
        for l in self.layers:
            i = l.create_state(k)
        self.converge_state(i, u, y, 10.)

        # Take a copy, and clean up the temporary state
        val = f(i)
        for l in self.layers:
            l.delete_state()
        return val

    def estimate_y(self, u, y):
        return self.converge(u, y, self.yest)

    def estimate_u(self, u, y):
        return self.converge(u, y, self.uest)

    def estimate_x(self, u, y):
        return self.converge(u, y, self.xest)
        #y = np.zeros((self.l_Y.m, k), dtype=np.float32) if self.l_Y else None

    def fit_classifier(self, classifier, y):
        classifier.fit(self.latent_layer(0).T, y)

    def xest(self, i=0):
        """
        Finds the last latent presentation for classification. The last or one
        before collision or CCA.
        """
        l = self.l_U
        while l.child and not isinstance(l.child, CCAModel):
            l = l.child
        # Should this be Xbar or the feedforward ?
        return l.X[i].var.get_value()

    def uest(self, i=0):
        return self.l_U.estimate(i).eval()

    def yest(self, i=0):
        return self.l_Y.estimate(i).eval()

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
