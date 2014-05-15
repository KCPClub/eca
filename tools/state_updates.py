#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


# Default values
l = 0.3
U_init = 0
Y_init = 0
X_init = 0
X2_init = 0
X3_init = 0
u = 10
y = 1


def update_accu_1(state):
    (U, X) = state if state else (U_init, X_init)
    U = 1*U + l*(u - 1*X)
    X = (1*U)
    return (U, X)


def update_accu_2(state):
    (U, X) = state if state else (U_init, X_init)
    U = 1*U + l*(u - 1*X)
    X = 1*X + l*(1*U)
    return (U, X)


def update_interp_2(state):
    (U, X) = state if state else (U_init, X_init)
    U = (1-l)*U + l*(u - 1 * X)
    X = (1-l)*X + l*(1 * U)
    return (U, X)


def update_accu_3(state):
    (U, X, X2) = state if state else (U_init, X_init, X2_init)
    U  = 1*U  + l*(u - 1*X)
    X  = 1*X  + l*(1*U - 1 * X2)
    X2 = 1*X2 + l*(1*X)
    return (U, X, X2)


def update_interp_3(state):
    (U, X, X2) = state if state else (U_init, X_init, X2_init)
    U = (1-l)*U + l*(u - 1 * X)
    X = (1-l)*X + l*(1 * U - 1 * X2)
    X2 = (1-l)*X2 + l*(1 * X)
    return (U, X, X2)


def update_two_way(state):
    (U, X, Y) = state if state else (U_init, X_init, Y_init)
    U = (1-l)*U + l*(u - 1 * X)
    Y = (1-l)*Y + l*(y - 1 * X)
    X = (1-l)*X + l*(1 * U + 1 * Y)
    return (U, X, Y)


def update_two_way_minus(state):
    (U, X, Y) = state if state else (U_init, X_init, Y_init)
    U = (1-l)*U + l*(u - 1 * X)
    Y = (1-l)*Y + l*(y - 1 * X)
    X = (1-l)*X + l*(1 * U - 1 * Y)
    return (U, X, Y)


def update_interp_4(state):
    (U, X, X2, X3) = state if state else (U_init, X_init, X2_init, X2_init)
    U = (1-l)*U + l*(u - 1 * X)
    X = (1-l)*X + l*(1 * U - 1 * X2)
    X2 = (1-l)*X2 + l*(1 * X - 1 * X3)
    X3 = (1-l)*X3 + l*(1 * X2)
    return (U, X, X2, X3)


def update_interp_1(state):
    (U, X) = state if state else (U_init, X_init)
    U = (1. - l) * U + l * (u - 1. * X)
    X = (1. * U)
    return (U, X)

updates = [
    #update_accu_one,
    update_two_way,
    update_two_way_minus,
    update_accu_1,
    update_accu_3,
    update_interp_1,
    update_interp_2,
    update_interp_3,
    update_interp_4,
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
        print '%.1f' % k,
    print
plt.show()

