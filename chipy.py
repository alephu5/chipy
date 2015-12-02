#!/usr/bin/python3

import argparse
import numpy as np
import sys
import os
from copy import deepcopy
import scipy.optimize
import importlib
import matplotlib.pyplot as plt

DESCRIPTION = """Automates common operations for chi-squared analysis used in
level 2 labs. The data should be presented as columns alternating between
measurements and their errors. The last pair of columns are assumed to
correspond to the dependent variable (a scalar), and those preceeding are an
independent vector."""


def parseargs():
    parser = argparse.ArgumentParser(DESCRIPTION)
    parser.add_argument(
        '--model_equation', type=importlib.import_module,
        default=lambda X, c: c, metavar='f(X)',
        help="""A string of the form [module].[function] which
                        represents the theoretical model that you expect between
                        the data. If a module is supplied without a function,
                        chi_squared analysis will be done for each function.

                        If no model is supplied, a null hypothesis is assumed,
                        unless the '--polyfit' argument is invoked.""")
    parser.add_argument('--polyfit', metavar='N', type=int,
                        help="""If model equation is not supplied, can iterate
                        throught polynomials of order 0-N and perform
                        chi-squared analysis for each curve.""")
    parser.add_argument('-i', type=str, default=sys.stdin,
                        metavar='INPUT',
                        help="""Columns of experimental data, assumed to be
                        alternating between experiment and errors.""")
    parser.add_argument('-o', type=open, default=sys.stdout,
                        metavar='PATH',
                        help='Path to output the results. Defaults to stdout.')
    parser.add_argument('--delimiter', type=str, default=',',
                        help="""Character used to separate columns.
                        Defaults to ','.""")
    return parser.parse_args()


def chisq(f, X, Y, sigma, *fparams):
    return (((f(X, *fparams) - Y) / sigma) ** 2).sum()


def chisq_err(f, X, Y, sigma, popt, tol=1e-8):
    chisQ = chisq(f, X, Y, sigma, popt)
    params = deepcopy(popt)
    while True:
        diffs = np.zeros(len(popt))
        for i in range(len(popt)):
            def g(p):
                args = deepcopy(params)
                args[i] = p
                return abs(chisq(f, X, Y, sigma, args) - (chisQ + 1))
            sol = scipy.optimize.minimize(g, popt[i], method='Powell').x
            diffs.itemset(i, params[i] - sol)
            params[i] = sol
        if (diffs**2).sum() < tol:
            return abs(params - popt)


def unpack_data(path, delimiter, filtr):
    # Measurements and errors are assumed to be alternating. The last
    # pair of columns corresponds to the dependent variable
    # while the preceeding are independent.

    # If filtr is True, values larger than the error are removed.
    raw = np.loadtxt(path, delimiter=delimiter, skiprows=1)
    meas = raw[:, ::2].transpose()
    err = raw[:, 1::2].transpose()
    if filtr:
        test = (abs(meas) >= err).prod(axis=0)
        meas = np.compress(test, meas, axis=1)
        err = np.compress(test, err, axis=1)
    return (meas[:-1], err[:-1]), (meas[-1], err[-1])


def main():
    args = parseargs()
    (X, Xerr), (Y, Yerr) = unpack_data(args.i, args.delimiter, True)
    f = args.model_equation
    popt, pcov = scipy.optimize.curve_fit(f, X, Y, sigma=Yerr)
    dof = len(Y) - len(popt)
    rchisq = chisq(f, X, Y, Yerr, popt) / dof
    rchisq_err = chisq_err(f, X, Y, Yerr, popt) / dof
    args.o.write('Optimal params: ' + str(popt) + os.linesep +
                 'Reduced chi-squared: ' + str(rchisq) + os.linesep +
                 'Having error :' + str(rchisq_err) + os.linesep)
    g = np.vectorize(f)
    plt.errorbar(X[0], Y, Yerr)
    plt.plot(X[0], g(X[0], popt))
    plt.show()


if __name__ == '__main__':
    main()
