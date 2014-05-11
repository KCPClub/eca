#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


def update_accu_one(U, X):
    U = 1*U + l*(u - 1*X)
    X = (1*U)
    return (U, X)


def update_accu_both(U, X):
    U = (1-l)*U + l*(u - 1*X)
    X = (1-l)*X + l*(1*U)
    return (U, X)


def update_interp_both(U, X):
    U = (1-l)*U + l*(u - 1*X)
    X = (1-l)*X + l*(1*U)
    return (U, X)


def update_interp_1(U, X):
    U = (1. - l) * U + l * (u - 1. * X)
    X = (1. * U)
    return (U, X)

updates = [update_accu_one,
           update_accu_both,
           update_interp_both,
           update_interp_1]

x = range(30)
for j in range(len(updates)):
    plt.subplot(len(updates), 1, j)
    l = 0.3
    U = 20
    X = 0
    u = 10
    y1 = []
    y2 = []
    for i in x:
        U, X = updates[j](U, X)
        print "U: %.1f, X: %.1f, u: %.2f" % (U, X, u)
        y1.append(U)
        y2.append(X)
    plt.plot(x, y1)
    plt.plot(x, y2)
plt.show()

