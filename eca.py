import numpy as np
from time import time
import theano
import theano.tensor as T
DEBUG_INFO = False
FLOATX = theano.config.floatX


def lerp(old, new, min_tau=0.0, en=None):
    """
    Return new interpolated value and a relative difference
    """
    diff = T.mean(T.sqr(new) - T.sqr(old), axis=1, keepdims=True)
    rel_diff = diff / (T.mean(T.sqr(old), axis=1, keepdims=True) + 1e-5)
    t = rel_diff * 20.
    t = T.where(t < 5, 5, t)
    t = T.where(t > 100, 100, t)
    t = t + min_tau
    if en is not None:
        lmbd = T.diagonal(en).dimshuffle(0, 'x') * (1. / t)
    else:
        lmbd = 1. / t
    return ((1 - lmbd) * old + lmbd * new,
            t, rel_diff)

def free_mem():
    from theano.sandbox.cuda import cuda_ndarray
    return cuda_ndarray.cuda_ndarray.mem_info()[0] / 1024 / 1024

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
        self.modulation = None
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
        self.X = {}
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

    def delete_state(self, id):
        # Discard the last one
        del self.X[id]
        del self.state_update_f[id]
        if id in self.model_update_f:
            del self.model_update_f[id]

    def create_state(self, k, id):
        if id in self.X.keys():
            self.delete_state(id)
        self.X[id] = State(self.n, k)

    def update_model(self, id, stiffness):
        if id in self.model_update_f:
            return self.model_update_f[id](stiffness)
        else:
            x = self.X[id]
            x_prev = self.parent.X[id]
            assert x_prev.k == x.k, "Sample size mismatch"
            assert x_prev.n == self.m, "Input dim mismatch"
            assert x.n == self.n, "Output dim mismatch"
            k = np.float32(x.k)
            # Modulate x
            if self.modulation is not None:
                x_ = x.var * T.as_tensor_variable(self.modulation)
            else:
                x_ = x.var
            (E_XU_new, t, d1) = lerp(self.E_XU,
                                     T.dot(x_, x_prev.var.T) / k)
            (E_XX_new, t, d2) = lerp(self.E_XX,
                                     T.dot(x_, x_.T) / k)
            E_XU_update = (self.E_XU, E_XU_new)
            E_XX_update = (self.E_XX, E_XX_new)
            b = 1.
            d = T.diagonal(E_XX_new)
            stiff = T.scalar('stiffnes', dtype=FLOATX)
            Q_new = theano.sandbox.linalg.ops.diag(b / T.where(d < stiff, stiff, d))
            Q_update = (self.Q, Q_new)

            # TODO: optional spatial neighborhood coupling
            phi_update = (self.phi, T.dot(Q_new, E_XU_new).T)

            d = T.maximum(T.max(d1), T.max(d2))
            self.model_update_f[id] = theano.function(
                inputs=[stiff],
                outputs=d,
                updates=[E_XU_update, E_XX_update, Q_update, phi_update])

            self.info('Model update between ' + self.name + ' and ' + self.parent.name)

            #self.info('Updating model, avg(diag(E_XX))=%.2f, ' % np.average(d) +
                      #'avg phi col norm %.2f' % np.average(np.sqrt(np.sum(np.square(self.phi),
                                                                           #axis=0))))
            # Call self now that we have the function set
            return self.update_model(id, stiffness)


    def update_state(self, id, input, min_tau=0.0):
        min_tau = 0.0
        if id in self.state_update_f:
            args = (min_tau,) if input is None else (min_tau, input)
            return self.state_update_f[id](*args)
        else:
            tau_in = T.scalar('min_tau', dtype=FLOATX)
            inputs = [tau_in]
            x = self.X[id]
            assert input is None or x.k == input.shape[1], "Sample size mismatch"

            # Get estimate of the state from layer above
            estimate = self.estimate(id)

            # Feedforward originates from previous layer's state or given input
            if input is None:
                feedforward = self.feedforward(id)
                has_missing_values = 0.0
            else:
                input_t = T.matrix('input', dtype=FLOATX)
                inputs += [input_t]
                missing_values = T.isnan(input_t)
                has_missing_values = T.any(missing_values)
                input_t = T.where(T.isnan(input_t), 0.0, input_t)
                feedforward = input_t

            self.info('Updating state[%s]: feedfwd from %s and estimate from %s' %
                      (str(id), self.parent.name if self.parent else 'input',
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
            if has_missing_values:
                new_value = T.where(missing_values, 0.0, new_value)

            (new_X, t, d) = lerp(x.var, new_value, tau_in)
            d = T.max(d)

            self.state_update_f[id] = theano.function(
                inputs=inputs,
                outputs=d,
                updates=[(x.var, new_X)])

            return self.update_state(id, input, min_tau)

    def estimate(self, id):
        """ Ask the child for feedback and apply nonlinearity """
        if not self.child:
            return 0.0
        fb = self.child.feedback(id, self)
        return self.nonlin_est(fb)

    def feedback(self, id, parent=None):
        return T.dot(self.phi, self.X[id].var)

    def feedforward(self, id):
        return T.dot(self.phi.T, self.parent.X[id].var)

    def set_modulation(self, mod):
        # Should be set before first model updates
        assert 'training' not in self.model_update_f
        self.modulation = mod

    def info(self, str):
        if DEBUG_INFO:
            print '%5s:' % self.name, str


class InputModel(Model):
    def __init__(self, name, n):
        super(InputModel, self).__init__(name,  n, None, None, True)

    def update_model(self, id, stiffness):
        return 0.0


class CollisionModel(Model):
    def __init__(self, name, n, (parent1, parent2), nonlin=None):
        relu = lambda x: T.where(x < 0, 0, x)
        self.u_side = Model(name + 'u', n, parent1, lambda x: x)
        self.y_side = Model(name + 'y', n, parent2, lambda x: x)

        # To make u and y update the same shared state, it must happen
        # simultaneously, so route u to ask y for its feedback as an estimate.
        self.u_side.estimate = lambda i: self.y_side.feedforward(i)
        self.u_side.custom_state_op = lambda fromu, fromy: nonlin(fromy + fromu)
        #self.u_side.custom_state_op = lambda fromu, fromy: nonlin(fromu)
        #self.u_side.custom_state_op = lambda fromu, fromy: (fromy + fromu)
        #self.u_side.custom_state_op = lambda fromu, fromy: T.sqrt(fromu * fromy + 0.1)

        # Disable state updates on the y side so that X is updated only once
        self.y_side.update_state = lambda i, input, min_tau: 0.0

        self.n = n
        self.X = {}
        self.name = name
        self.phi = self.u_side.phi  # For visualization

    def update_model(self, id, stiffness):
        d1 = self.u_side.update_model(id, stiffness)
        d2 = self.y_side.update_model(id, stiffness)
        return np.maximum(d1, d2)

    def update_state(self, id, input, min_tau):
        assert False, 'should be never called'

    def delete_state(self, id):
        del self.X[id].var
        del self.X[id]
        del self.u_side.state_update_f[id]
        assert self.y_side.state_update_f == {}
        if id in self.u_side.model_update_f:
            del self.u_side.model_update_f[id]
        if id in self.y_side.model_update_f:
            del self.y_side.model_update_f[id]

    def create_state(self, k, id):
        if id in self.X.keys():
            self.delete_state(id)
        self.X[id] = State(self.n, k)
        self.u_side.X = self.X
        self.y_side.X = self.X

    def get_feedback(self, id, parent=None):
        assert False, "should not be called"


class CCAModel(Model):
    def __init__(self, name, (m, n)):
        self.E_ZZ = []
        super(CCAModel, self).__init__(name, (m, n), nonlin=None,
                                       identity=False)
        # Phi of this layer is not in use, mark it as nan
        self.phi = np.zeros((1, 1))

    def update_model(self, id, stiffness):
        """ There is no global model state for CCA layer, instead, everything
        happens in update_state
        """
        return 0.0

    def create_state(self, k):
        # Create a stack of E_ZZ in addition to base class implementation
        self.E_ZZ.append(State(1, 1))
        return super(CCAModel, self).create_state(k)

    def delete_state(self, id):
        del self.E_ZZ[id]
        return super(CCAModel, self).delete_state(id)

    def update_state(self, id, input, min_tau):
        z = self.X[id]
        E_ZZ = self.E_ZZ[id]
        assert input is None, "CCA state cannot use input"
        assert self.child is None, 'CCA cannot have children'
        assert len(self.parent) == 2, 'CCA should have exactly 2 parents'
        assert z.n == self.n, "Output dim mismatch"

        # Update state-specific E_ZZ and calculate q for z update
        (E_ZZ.value, di) = lerp(E_ZZ.value,
                                np.dot(z.value, z.value.T) / z.k)
        assert E_ZZ.value.shape == (1, 1), 'E_ZZ is not a scalar!?'
        b = 1.
        q = b / np.max([0.05, E_ZZ.value[0, 0]])

        # Update z
        [x1, x2] = [self.parent[j].X[id].value for j in (0, 1)]
        new_value = q * np.sum(x1 * x2, axis=0)

        (z.value, dz) = lerp(z.value, new_value)

        self.E_XU = None
        return np.max((np.abs(di), np.average(np.abs(dz))))

    def get_feedback(self, id, parent=None):
        assert parent is not None, 'CCA needs parent'
        assert len(self.parent) == 2, 'CCA should have exactly 2 parents'
        # XXX: Uncomment the following to avoid signal propagation
        # back from CCA layer
        #return np.zeros(())

        z = self.X[id]
        # Find the other branch connecting to this z
        x_other = list(set(self.parent) - set([parent]))[0]

        # This is roughly: x_u_delta = self.Q * x_y * z
        phi = self.Q * x_other.X[id].value
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
        #self.l_Y.nonlin_est = lambda x: T.nnet.softmax(x.T).T

        # Then add connecting layer to connect u and y branches:
        # ... -> X_uN -> Z
        # ... -> Y    --'
        print 'Creating collision layer with parents:', layers[-2].name, layers[-1].name
        cl = CollisionModel('Z', n_layers[-1], (layers[-2], layers[-1]), nonlin)
        layers += [cl]

    def update(self, u, y, stiffness):
        """
        Performs update round-trip from u to X and back.
        """
        # Create default state now when we know the sample count
        if len(self.l_U.X) == 0 or self.l_U.X['training'].k != u.shape[1]:
            for l in self.layers:
                l.create_state(u.shape[1], 'training')

        self.update_states('training', u, y)
        d = self.update_models('training', stiffness)
        return d

    def update_models(self, id, stiffness):
        max_delta = 0.0
        for l in self.layers:
            d = l.update_model(id, stiffness)
            max_delta = np.maximum(d, max_delta)
        return max_delta

    def update_states(self, id, u, y, min_tau=0.0):
        max_diff = 0.0

        # Loop over net inputs
        for (x, val) in zip([self.l_U, self.l_Y], [u, y]):
            # Loop through the chain of layers
            while x:
                d = x.update_state(id, val, min_tau)
                x = x.child
                val = None  # Only the first layer will have value in
                max_diff = max(d, max_diff)
        return max_diff

    def converge_state(self, id, u, y, min_tau=0.0):
        max_delta = 1.0
        iter = 0
        (delta_limit, time_limit, iter_limit) = (1e-3, 20, 200)
        t_start = time()
        # Convergence condition, pretty arbitrary for now
        while max_delta > delta_limit and time() - t_start < time_limit:
            max_delta = self.update_states(id, u, y, min_tau)
            iter += 1
            if iter >= iter_limit:
                break
        if False:
            print 'Converged in', "%.1f" % (time() - t_start), 's,', iter,
            print 'iters, delta %.4f' % max_delta,
            print 'Limits: i:', iter_limit, 't:', time_limit, 'd:', delta_limit

    def converge(self, id, u, y, f=None):
        k = u.shape[1]
        for l in self.layers:
            l.create_state(k, id)
        self.converge_state(id, u, y)

        if f:
            return f(id)

    def estimate_y(self, u, y, id):
        return self.converge(id, u, y, self.yest)

    def estimate_u(self, u, y, id):
        return self.converge(id, u, y, self.uest)

    def estimate_x(self, u, y, id):
        return self.converge(id, u, y, self.xest)

    def xest(self, id='training', no_eval=False):
        """
        Finds the last latent presentation for classification. The last or one
        before collision or CCA.
        """
        l = self.l_U
        while l.child and not isinstance(l.child, CCAModel):
            l = l.child
        # Should this be Xbar or the feedforward ?
        v = l.X[id].var
        return v if no_eval else v.eval()

    def uest(self, id='training', no_eval=False):
        v = self.l_U.estimate(id)
        return v if no_eval else v.eval()

    def yest(self, id='training', no_eval=False):
        v = self.l_Y.estimate(id)
        return v if no_eval else v.eval()

    def reconst_err_u(self, u, y, id):
        u_est = self.estimate_u(u, y, id)
        return np.average(np.square(u_est - u), axis=0)

    def first_phi(self):
        # index is omitted for now, and the lowest layer is plotted
        return self.l_U.child.phi.get_value()

    def variance(self, states=None):
        f = lambda l: (l.name, l.X['training'].variance())
        return map(f, self.layers)

    def energy(self, id='training'):
        f = lambda l: (l.name, np.average(l.X[id].energy()))
        return map(f, self.layers)

    def avg_levels(self):
        f = lambda l: (l.name, np.linalg.norm(np.average(l.X['training'].var.get_value(), axis=1)))
        return map(f, self.layers)

    def phi_norms(self):
        f = lambda l: (l.name, (np.linalg.norm(l.phi.get_value(), axis=0)))
        return map(f, list(set(self.layers) - set([self.l_U, self.l_Y])))
