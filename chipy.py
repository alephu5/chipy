#!/usr/bin/python3

import argparse
import numpy as np
import sys
import os
from copy import deepcopy
import scipy.optimize
from importlib import import_module
import matplotlib.pyplot as plt
import matplotlib

DESCRIPTION = """Automates common operations for chi-squared analysis used in
level 2 labs. The data should be presented as columns alternating between
measurements and their errors. The last pair of columns are assumed to
correspond to the dependent variable (a scalar), and those preceeding are an
independent vector."""


def parseargs():
    parser = argparse.ArgumentParser(DESCRIPTION)
    parser.add_argument(
        '-f', type=load_attribute,
        default=lambda X, c: c, metavar='f(X)',
        help="""A string of the form [module].[function] which
                represents the theoretical model that you expect between
                the data.

                If no model is supplied, a null hypothesis is assumed""")
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
    parser.add_argument('-l', type=str, metavar='Layout file',
                        help="""Path to python script containing matplotlib
                        layout parameters.""")
    return parser.parse_args()


def chisq(f, X, Y, sigma, *fparams):
    return (((f(X, *fparams) - Y) / sigma) ** 2).sum()


def chisq_err(f, X, Y, sigma, popt, tol=1e-8):
    chisQ = chisq(f, X, Y, sigma, *popt)
    params = deepcopy(popt)
    while True:
        diffs = np.zeros(len(popt))
        for i in range(len(popt)):
            def g(p):
                args = deepcopy(params)
                args[i] = p
                return abs(chisq(f, X, Y, sigma, *args) - (chisQ + 1))
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

    if meas.shape[0] == 2:
        A = (meas[:-1].ravel(), err[:-1].ravel())
        return A, (meas[-1], err[-1])
    else:
        return (meas[:-1], err[:-1]), (meas[-1], err[-1])


def get_headings(path, delimiter):
    with open(path) as f:
        headings = f.readline()
        headings = headings.split(delimiter)
    return headings


def load_attribute(astring):
    m, a = astring.split('.')
    mdl = import_module(m)
    return getattr(mdl, a)


def min_chisq(f, X, Y, Yerr):
    if isinstance(f, np.poly1d):
        popt = f.c
        dof = len(Y) - len(popt)
        rchisq = chisq(f, X, Y, Yerr) / dof
        rchisq_err = 0
    else:
        popt, pcov = scipy.optimize.curve_fit(f, X, Y, sigma=Yerr)
        dof = len(Y) - len(popt)
        rchisq = chisq(f, X, Y, Yerr, *popt) / dof
        rchisq_err = chisq_err(f, X, Y, Yerr, popt) / dof
    return popt, pcov, rchisq, rchisq_err


def main():
    args = parseargs()
    if args.l:
        font = load_attribute(args.l + '.font')
        matplotlib.rc('font', **font)
    (X, Xerr), (Y, Yerr) = unpack_data(args.i, args.delimiter, True)
    headings = get_headings(args.i, args.delimiter)
    f = args.f
    popt, pcov, rchisq, rchisq_err = min_chisq(f, X, Y, Yerr)
    args.o.write('Optimal params: ' + str(popt) + os.linesep +
                 'With uncertainty: ' + str(np.sqrt(pcov)) + os.linesep +
                 'Reduced chi-squared: ' + str(rchisq) + os.linesep +
                 'Having error :' + str(rchisq_err) + os.linesep)

    g = np.vectorize(f)
    plt.errorbar(X, Y, Yerr, label='Measurements', lw=0, marker='x',
                 ms=3, mew=5, capsize=5, mec='green', mfc='red')
    plt.plot(X, g(X, *popt), label='Model', lw=2, color='magenta')
    plt.xlabel(headings[0])
    plt.ylabel(headings[-2])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
