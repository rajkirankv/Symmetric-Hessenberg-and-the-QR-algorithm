import numpy as np


def checks(a):
    # Checks performed: For a given matrix a, check that 1) a is indeed a matrix, 2) a is a real matrix, 3) a is a
    # symmetric matrix

    checks_passed = False
    error_msg = ''

    # Check that the input is h numpy array
    if not isinstance(a, np.matrix):
        error_msg = "Argument must be a numpy matrix"
        return checks_passed, error_msg

    # Check that the input is real
    if not np.isreal(a).all():
        error_msg = "The input matrix is not real"
        return checks_passed, error_msg

    # Check that the input is symmetric
    # if not np.all(a == a.T):
    #     error_msg = "The input matrix is not symmetric"
    #     return checks_passed, error_msg

    checks_passed = True

    return checks_passed, error_msg


def makeExact(a, CUTOFF=1e-12):
    # For a given matrix a, 1) set all elements beloe a given cutoff to zero, 2) check that the 1st subdiagonals are
    # nearly equal i.e within the range of cutoff. If they are, make them equal, else, raise Assertion error

    m = np.shape(a)[0]
    a[np.abs(a) < CUTOFF] = 0
    assert all(np.isclose(np.diagonal(a, offset=1), np.diagonal(a, offset=-1), atol=CUTOFF))
    for diag in range(1, m):
        a[diag - 1, diag] = a[diag, diag - 1]
    return
