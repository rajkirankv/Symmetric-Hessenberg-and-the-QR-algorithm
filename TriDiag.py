# Compute the symmertric hessenberg i.e the tridiagonal form of a given real, symmetric matrix

import scipy.linalg as la, numpy as np
import copy

import Utils


def tridiag(a, exact=1):
    CUTOFF = 1e-12

    flag, msg = Utils.checks(a)
    if flag is False:
        raise ValueError(msg)

    h = copy.deepcopy(a)  # This is the hessenberg matrix to be returned
    m = np.shape(h)[0]

    for k in range(m - 2):
        x = h[k + 1:m, k]
        e1 = np.asmatrix(np.zeros(np.size(x))).T
        e1[0] = 1
        v = np.sign(x[0, 0]) * np.linalg.norm(x) * e1 + x
        v = v / np.linalg.norm(v)
        h[k + 1:m, k:m] -= 2 * v * (v.T * h[k + 1:m, k:m])
        h[0:m, k + 1:m] -= 2 * (h[0:m, k + 1:m] * v) * v.T

    if exact == 1:
        Utils.makeExact(h, CUTOFF)

    return h


def main():
    a = np.asmatrix(la.hilbert(4))
    h = tridiag(a)
    h_std = la.hessenberg(a)
    print("Solution:")
    print(h)
    print('\n')
    print("Standard Solution:")
    print(h_std)


if __name__ == '__main__':
    main()

##
# Solution:
# [[  1.00000000e+00  -6.50854140e-01   0.00000000e+00   0.00000000e+00]
#  [ -6.50854140e-01   6.50585480e-01   6.39118800e-02   0.00000000e+00]
#  [  0.00000000e+00   6.39118800e-02   2.53201434e-02  -1.16520804e-03]
#  [  0.00000000e+00   0.00000000e+00  -1.16520804e-03   2.84852680e-04]]
#
#
# Standard Solution:
# [[  1.00000000e+00  -6.50854140e-01  -1.18717050e-16  -8.82852051e-17]
#  [ -6.50854140e-01   6.50585480e-01   6.39118800e-02   4.51346645e-17]
#  [  0.00000000e+00   6.39118800e-02   2.53201434e-02  -1.16520804e-03]
#  [  0.00000000e+00   0.00000000e+00  -1.16520804e-03   2.84852680e-04]]
##
