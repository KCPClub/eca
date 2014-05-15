#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


# Default values
l = 0.3
U1_0 = 0
Y1_0 = 0
X1_0 = 0
X2_0 = 0
X3_0 = 0
phi_0 = 1.
u = 10
y = 1

def update_accu_1(state):
    (U, X) = state if state else (U1_0, X2_0)
    U = 1*U + l*(u - 1*X)
    X = (1*U)
    return (U, X)


def update_accu_both(state):
    (U, X) = state if state else (U1_0, X1_0)
    U = 1*U + l*(u - 1*X)
    X = 1*X + l*(1*U)
    return (U, X)


def update_accu_all_3(state):
    (U1, X1, X2, phi1, phi2) = state if state else (U1_0, X1_0, X2_0, phi_0,
                                                    phi_0)
    phi1 = lerp(phi1, q(X1) * X1 * U1)
    phi2 = lerp(phi2, q(X2) * X2 * X1)
    U1 += l * (u - phi1*X1)
    X1 += l * (phi1 * U1 - phi2 * X2)
    X2 += l * (phi2 * X1)
    return (U1, X1, X2, phi1, phi2)


def lerp(old, new):
    return (1-l) * old + l * new


def q(X):
    return 1. / max((X * X), 1e-10)


def update_interp_both(state):
    (U, X, phi) = state if state else (U1_0, X1_0, phi_0)
    phi = lerp(phi, q(X) * X*U)
    U = lerp(U, u - phi * X)
    X = lerp(X,     phi * U)
    return (U, X, phi)


def update_interp_all_3(state):
    (U1, X1, X2, phi1, phi2) = state if state else (U1_0, X1_0, X2_0,
                                                    phi_0, phi_0)
    phi1 = lerp(phi1, q(X1) * X1*U1)
    phi2 = lerp(phi2, q(X2) * X2*X1)

    U1 = lerp(U1,         u - phi1 * X1)
    X1 = lerp(X1, phi1 * U1 - phi2 * X2)
    X2 = lerp(X2, phi2 * X1)
    return (U1, X1, X2, phi1, phi2)


def update_interp_all_4(s):
    U1, X1, X2, X3, phi1, phi2, phi3 = s if s else (U1_0, X1_0, X2_0,
                                                    X2_0, phi_0,
                                                    phi_0, phi_0)
    phi1 = lerp(phi1, q(X1) * X1*U1)
    phi2 = lerp(phi2, q(X2) * X2*X1)
    phi3 = lerp(phi3, q(X3) * X3*X2)

    U1 = lerp(U1,         u - phi1 * X1)
    X1 = lerp(X1, phi1 * U1 - phi2 * X2)
    X2 = lerp(X2, phi2 * X1 - phi3 * X3)
    X3 = lerp(X3, phi3 * X2)
    return (U1, X1, X2, X3, phi1, phi2, phi3)


def update_two_way(state, sign=1., a=1.):
    (U, X, Y, phiU, phiY) = state if state else (U1_0, X1_0, Y1_0,
                                                 phi_0, phi_0)
    phiU = lerp(phiU, q(X) * X * U)
    phiY = lerp(phiY, q(X) * X * Y)
    U = lerp(U,        u - phiU * X)
    Y = lerp(Y,        y - phiY * X)
    X = lerp(X, (phiU * U + phiY * Y * sign) * a)
    return (U, X, Y, phiU, phiY)


def update_two_way_minus(state):
    return update_two_way(state, -1)


def update_two_way_avg(state):
    return update_two_way(state, a=0.5)


def update_two_way_avg_minus(state):
    return update_two_way(state, sign=-1, a=0.5)


def update_interp_other(state):
    (U, X) = state if state else (U1_0, X1_0)
    U = lerp(U, u - 1. * X)
    X = (1. * U)
    return (U, X)

updates = [
    #update_accu_1,
    #update_accu_both,
    update_accu_all_3,
    update_two_way,
    update_two_way_avg,
    update_two_way_avg_minus,
    update_two_way_minus,
    update_interp_other,
    update_interp_both,
    update_interp_all_3,
    update_interp_all_4,
]

for j in range(len(updates)):
    plt.subplot(len(updates), 1, j)
    plt.title(updates[j].func_name)
    print updates[j].func_name

    t = range(30)
    state = None
    states = []
    for i in t:
        state = updates[j](state)
        states += [state]
    for i in np.array(states).T:
        plt.plot(t, i)
    print 'u: %.1f' % u, 'state:',
    for k in state:
        print '%.2f' % k,
    print
plt.show()

