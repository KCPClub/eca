from time import time

import numpy as np
import theano
import theano.tensor as T
DEBUG_INFO = False
from theano.ifelse import ifelse
PRINT_CONVERGENCE = False
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


class Signal(object):
    """ Object that represents any kind of state U, X, X_y, z, ...
    """
    def __init__(self, n, k, name):
        rng = np.random.RandomState(0)
        self.var = theano.shared(np.float32(rng.uniform(size=(n, k))), name=name)
        self.k = k
        self.n = n
        self.modulation = None

    def set_modulation(self, mod):
        assert self.modulation is None
        self.modulation = mod

    def variance(self):
        return np.average(np.log(np.var(self.var.get_value(), axis=1)))

    def energy(self):
        return np.average(np.square(self.var.get_value()), axis=1)


class Layer(object):
    def __init__(self, name, n, prev, nonlin, identity=False):
        m = n if prev is None else prev.n
        rng = np.random.RandomState(0)
        self.n = n
        self.m = m
        self.nonlin = nonlin
        # Nonlinearity applied to the estimate coming from next layer
        self.nonlin_est = lambda x: x
        rand_init = np.float32(rng.uniform(size=(n, m)) - 0.5)
        self.E_XU = theano.shared(rand_init, name='E_XU')
        self.E_XX = theano.shared(np.identity(n, dtype=FLOATX), name='E_XX')
        self.Q = theano.shared(np.identity(n, dtype=FLOATX), name='Q')
        self.phi = theano.shared(rand_init.T, name='phi')
        self.name = name
        self.signal_key = name
        self.missing_values = None
        self.custom_state_op = None
        self.prev = prev
        self.next = None

        if self.prev:
            assert prev.next is None
            prev.next = self

    def compile_adapt_f(self, signals):
        x = self.signal(signals)
        x_prev = self.prev.signal(signals)
        assert x_prev.k == x.k, "Sample size mismatch"
        assert x_prev.n == self.m, "Input dim mismatch"
        assert x.n == self.n, "Output dim mismatch"
        k = np.float32(x.k)
        # Modulate x
        if x.modulation is not None:
            x_ = x.var * T.as_tensor_variable(x.modulation)
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

        self.info('Compile layer update between: ' + self.name + ' and ' + self.prev.name)
        d = T.maximum(T.max(d1), T.max(d2))
        return theano.function(
            inputs=[stiff],
            outputs=d,
            updates=[E_XU_update, E_XX_update, Q_update, phi_update])

    def signal(self, signals):
        key = self.signal_key
        if key not in signals.signal:
            signals.signal[key] = Signal(self.n, signals.k, self.name)
        return signals.signal[key]

    def compile_prop_f(self, signals, has_input, min_tau=0.0):
        tau_in = T.scalar('min_tau', dtype=FLOATX)
        inputs = [tau_in]
        x = self.signal(signals)

        # Get estimate of the state from layer above
        estimate = self.estimate(signals)

        # Feedforward originates from previous layer's state or given input
        if not has_input:
            feedforward = self.feedforward(signals)
            has_nans = T.as_tensor_variable(0)
            nans = 0.0
        else:
            input_t = T.matrix('input', dtype=FLOATX)
            inputs += [input_t]
            nans = T.isnan(input_t)
            has_nans = T.any(nans)
            input_t = T.where(T.isnan(input_t), 0.0, input_t)
            feedforward = input_t

        self.info('Compiling propagation: %6s %4s %6s' %
                  (self.prev.name + ' ->' if self.prev else 'u/y ->',
                   self.name,
                   '<- ' + self.next.name if self.next else ""))

        # Apply nonlinearity to feedforward path only
        if self.nonlin:
            feedforward = self.nonlin(feedforward)

        if self.custom_state_op:
            new_value = self.custom_state_op(feedforward, estimate)
        else:
            new_value = feedforward - estimate

        # If predicting missing values, force them to zero in residual so
        # that they don't influence learning
        new_value = ifelse(has_nans, T.where(nans, 0.0, new_value), new_value)

        (new_X, t, d) = lerp(x.var, new_value, tau_in)
        d = T.max(d)

        return theano.function(inputs=inputs,
                               outputs=d,
                               updates=[(x.var, new_X)])

    def estimate(self, signals):
        """ Ask the next for feedback and apply nonlinearity """
        if not self.next:
            return 0.0
        return self.nonlin_est(self.next.feedback(signals))

    def feedback(self, signals):
        x = self.signal(signals)
        return T.dot(self.phi, x.var)

    def feedforward(self, signals):
        x = self.prev.signal(signals)
        return T.dot(self.phi.T, x.var)

    def info(self, str):
        if DEBUG_INFO:
            print '%5s:' % self.name, str


class InputLayer(Layer):
    def __init__(self, name, n):
        super(InputLayer, self).__init__(name,  n, None, None, True)

    def compile_adapt_f(self, signals):
        return lambda stiff: 0.0


class CollisionLayer(Layer):
    def __init__(self, name, n, (prev1, prev2), nonlin=None):
        self.u_side = Layer(name + 'u', n, prev1, lambda x: x)
        self.y_side = Layer(name + 'y', n, prev2, lambda x: x)
        self.u_side.signal_key = name
        self.y_side.signal_key = name
        self.signal_key = name
        self.name = name
        self.merge_op = lambda fromu, fromy: nonlin(fromu + fromy)
        # TODO: figure how to expose bot u_side and y_side phi
        self.phi = self.u_side.phi

        # To make u and y update the same shared state, it must happen
        # simultaneously, so route u to ask y for its feedback as an estimate.
        self.u_side.estimate = lambda i: self.y_side.feedforward(i)
        self.u_side.custom_state_op = lambda fromu, fromy: nonlin(fromy + fromu)
        #self.u_side.custom_state_op = lambda fromu, fromy: nonlin(fromu)
        #self.u_side.custom_state_op = lambda fromu, fromy: (fromy + fromu)
        #self.u_side.custom_state_op = lambda fromu, fromy: T.sqrt(fromu * fromy + 0.1)

        # Disable state updates on the y side so that X is updated only once
        self.y_side.update_state = lambda i, input, min_tau: 0.0

    def create_signal(self, signals):
        return Signal(self, self.n, signals.k, self.name)

    def update_state(self, id, input, min_tau):
        assert False, 'should be never called'

    def compile_adapt_f(self, signals):
        assert False, "should not be called"


class CCALayer(Layer):
    def __init__(self, name, (m, n)):
        self.E_ZZ = []
        super(CCALayer, self).__init__(name, (m, n), nonlin=None,
                                       identity=False)
        # Phi of this layer is not in use, mark it as nan
        self.phi = np.zeros((1, 1))

    # TODO: Update
    def update_state(self, id, input, min_tau):
        z = self.X[id]
        E_ZZ = self.E_ZZ[id]
        assert input is None, "CCA state cannot use input"
        assert self.next is None, 'CCA cannot have next items'
        assert len(self.prev) == 2, 'CCA should have exactly 2 prevs'
        assert z.n == self.n, "Output dim mismatch"

        # Update state-specific E_ZZ and calculate q for z update
        (E_ZZ.value, di) = lerp(E_ZZ.value,
                                np.dot(z.value, z.value.T) / z.k)
        assert E_ZZ.value.shape == (1, 1), 'E_ZZ is not a scalar!?'
        b = 1.
        q = b / np.max([0.05, E_ZZ.value[0, 0]])

        # Update z
        [x1, x2] = [self.prev[j].X[id].value for j in (0, 1)]
        new_value = q * np.sum(x1 * x2, axis=0)

        (z.value, dz) = lerp(z.value, new_value)

        self.E_XU = None
        return np.max((np.abs(di), np.average(np.abs(dz))))

    # TODO: Update
    def feedback(self, id, prev=None):
        assert prev is not None, 'CCA needs prev'
        assert len(self.prev) == 2, 'CCA should have exactly 2 prevs'
        # XXX: Uncomment the following to avoid signal propagation
        # back from CCA layer
        #return np.zeros(())

        z = self.X[id]
        # Find the other branch connecting to this z
        x_other = list(set(self.prev) - set([prev]))[0]

        # This is roughly: x_u_delta = self.Q * x_y * z
        phi = self.Q * x_other.X[id].value
        return phi * z.value


class ECA(object):
    """
    Constructs chains of layers connected together. Each layer contains phi
    matrix and corresponding signal vector X.

    Typical loop u -- U -- X would represented by two of these layers; one for
    U and another one for X.

    U branch would look something like this, where --- indicates signal
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
        self.U = InputLayer('U', n_u)
        self.layers = layers = [self.U]
        #self.U.nonlin_est = lambda x: T.eq(x, T.max(x, axis=0, keepdims=True))
        #self.U.nonlin_est = lambda x : T.nnet.sigmoid(x)
        #self.U.nonlin_est = nonlin

        # Then the consecutive layers U -> X_u1 -> X_u2 -> ...
        n_ulayers = n_layers[:-1] if n_y else n_layers
        for (i, n) in enumerate(n_ulayers):
            m = Layer("X_u%d" % (i + 1), n, layers[-1], nonlin)
            layers.append(m)

        self.Y = None
        self.collision = None

        # No supervised y signal => all done!
        if not n_y:
            return

        # 2. Create layers for y branch if still here
        # Keep y branch shorter by only taking the last layer dimension
        print 'Creating layer (%d) for classification' % n_y

        # First the input layer Y
        self.Y = InputLayer('Y', n_y)
        layers += [self.Y]
        #self.Y.nonlin_est = lambda x: T.eq(x, T.max(x, axis=0, keepdims=True))
        #self.Y.nonlin_est = lambda x: T.nnet.softmax(x.T).T

        # Then add connecting layer to connect u and y branches:
        # ... -> X_uN -> Z
        # ... -> Y    --'
        print 'Creating collision layer with next:', layers[-2].name, layers[-1].name
        cl = CollisionLayer('Z', n_layers[-1], (layers[-2], layers[-1]), nonlin)
        layers += [cl]

    def new_signals(self, k):
        return Signals(k, self)

    def first_phi(self):
        # index is omitted for now, and the lowest layer is plotted
        return self.U.next.phi.get_value()

    def phi_norms(self):
        f = lambda l: (l.name, (np.linalg.norm(l.phi.get_value(), axis=0)))
        return map(f, list(set(self.layers) - set([self.U, self.Y])))


class Signals(object):
    def __init__(self, k, eca):
        self.mdl = eca
        self.k = k
        self.adaptf = {}
        self.propf = {}
        self.signal = {}
        self.name = None

        for l in [self.mdl.U, self.mdl.Y]:
            is_input = True
            while l:
                self.propf[l.name] = l.compile_prop_f(self, is_input)
                is_input = False
                l = l.next

    def _iter_layers(self):
        for l in [self.mdl.U, self.mdl.Y]:
            while l:
                yield l
                l = l.next

    def _compile_adapt_fs(self):
        for l in self._iter_layers():
            self.adaptf[l.name] = l.compile_adapt_f(self)

        #for l in [self.mdl.U, self.mdl.Y]:
            #while l:
                #self.adaptf[l.name] = l.compile_adapt_f(self)
                #l = l.next

    def adapt_layers(self, stiffness):
        if self.adaptf == {}:
            self._compile_adapt_fs()

        for l in self._iter_layers():
            self.adaptf[l.name](stiffness)

        #for l in [self.mdl.U, self.mdl.Y]:
            #while l:
                #self.adaptf[l.name](stiffness)
                #l = l.next

    def propagate(self, u, y, min_tau=0.0):
        assert u is None or self.k == u.shape[1], "Sample size mismatch"
        assert y is None or self.k == y.shape[1], "Sample size mismatch"
        d = 0.0
        for (l, inp) in zip([self.mdl.U, self.mdl.Y], [u, y]):
            while l:
                f = self.propf[l.name]
                args = [min_tau] + ([] if inp is None else [inp])
                d = max(d, f(*args))
                l = l.next
                inp = None
        return d

    def converge(self, u, y, min_tau=0.0, f=None):
        t = 20
        (d_limit, t_limit, i_limit) = (1e-3, time() + t, 200)
        d, i, = np.inf, 0
        while d > d_limit and time() < t_limit and i < i_limit:
            d = self.propagate(u, y, min_tau)
            i += 1
        if PRINT_CONVERGENCE:
            print 'Converged in', "%.1f" % (time() - t_limit + t), 's,',
            print i, 'iters, delta %.4f' % d,
            print 'Limits: i:', i_limit, 't:', t, 'd:', d_limit
        return f() if f else None

    def estimate_y(self, u, y):
        return self.converge(u, y, f=self.yest)

    def estimate_u(self, u, y):
        return self.converge(u, y, f=self.uest)

    def estimate_x(self, u, y):
        return self.converge(u, y, f=self.xest)

    def xest(self, no_eval=False):
        # TODO: Might not be reliable, fix.
        l = self.mdl.U
        while l.next and not isinstance(l.next, CCALayer):
            l = l.next
        # Should this be Xbar or the feedforward ?
        v = l.signal(self).var
        return v if no_eval else v.eval()

    def uest(self, no_eval=False):
        v = self.mdl.U.estimate(self)
        return v if no_eval else v.eval()

    def yest(self, no_eval=False):
        v = self.mdl.Y.estimate(self)
        return v if no_eval else v.eval()

    def reconst_err_u(self, u, y):
        u_est = self.estimate_u(u, y)
        return np.average(np.square(u_est - u), axis=0)

    def first_phi(self):
        # index is omitted for now, and the lowest layer is plotted
        return self.mdl.U.next.phi.get_value()

    def variance(self, states=None):
        f = lambda l: (l.name, l.signal(self).variance())
        return map(f, self.mdl.layers)

    def energy(self):
        f = lambda l: (l.name, np.average(l.signal(self).energy()))
        return map(f, self.mdl.layers)

    def avg_levels(self):
        f = lambda l: (l.name, np.linalg.norm(np.average(l.signal(self).var.get_value(), axis=1)))
        return map(f, self.mdl.layers)

    def phi_norms(self):
        f = lambda l: (l.name, (np.linalg.norm(l.phi.get_value(), axis=0)))
        return map(f, list(set(self.mdl.layers) - set([self.mdl.U, self.mdl.Y])))

