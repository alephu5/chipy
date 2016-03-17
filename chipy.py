#!/usr/bin/python3

import argparse
import numpy as np
import sys
from copy import deepcopy
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib
from fileops import load_attribute, unpack_data, write_report

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
    parser.add_argument('-i', type=str, default=sys.stdin, nargs='*',
                        metavar='INPUT',
                        help="""Columns of experimental data, assumed to be
                        alternating between experiment and errors.""")
    parser.add_argument('-c', type=str, metavar='CONTROL',
                        help="""Uses the argument as a control condition. All
                        fitted parameters are subtracted from this when
                        reporting the final result and the errors are
                        propagated.""")
    parser.add_argument('--delimiter', type=str, default=',',
                        help="""Character used to separate columns.
                        Defaults to ','.""")
    parser.add_argument('-l', type=str, metavar='LAYOUT FILE',
                        help="""Path to python script containing matplotlib
                        layout parameters.""")
    parser.add_argument('-s', type=str, metavar='SAVE GRAPH',
                        help="""Automatically save graph in the specified
                        directory.""")
    parser.add_argument('-r', type=str, metavar='REPORT',
                        help="""Writes results to the file specified, using
                        procedures defined in the function file.""")
    parser.add_argument('--show', dest='show', action='store_true',
                        help="Show a graph.")
    parser.add_argument('--no-show', dest='show', action='store_false',
                        help="Don't show a graph.")
    parser.add_argument('--split', type=int, help="""Splits the file into
                       seperate pieces; grouping them by values in the column
                       specified. The first column is 0.""", default=-1)
    parser.set_defaults(show=True)
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


def min_chisq(f, X, Y, Yerr):
    if isinstance(f, np.poly1d):
        popt = f.c
        dof = len(Y) - len(popt)
        rchisq = chisq(f, X, Y, Yerr) / dof
        # rchisq_err = 0
    else:
        popt, pcov = scipy.optimize.curve_fit(f, X, Y, sigma=Yerr)
        dof = len(Y) - len(popt)
        rchisq = chisq(f, X, Y, Yerr, *popt) / dof
        # rchisq_err = chisq_err(f, X, Y, Yerr, popt) / dof
        # Removed because of questionable accuracy for multiple parameters.
        uncerts = np.sqrt(pcov.diagonal())
    return popt, uncerts, rchisq  # , rchisq_err


def test_hyp(f_class, data):
    f = f_class.f
    (X, Xerr), (Y, Yerr) = data
    X = f_class.pre_process(X)
    return min_chisq(f, X, Y, Yerr)


def update_graphs(f_class, data_name, data, popt, fig, ax_main, ax_res):
    f = f_class.f
    (X, Xerr), (Y, Yerr) = data

    g = np.vectorize(f)
    S = np.sort(X)
    model = g(S, *popt)

    ax_main.errorbar(
        X, Y, Yerr, label='Measured ' + data_name, lw=0, marker='x',
        ms=0.1, mew=2, capsize=1)
    ax_main.plot(S, model, label='Model ' + data_name, lw=2)
    ax_res.errorbar(X, Y - g(X, *popt), Yerr, label=data_name, lw=0.1)


def main():
    args = parseargs()
    if args.l:
        # Load layout
        font = load_attribute(args.l + '.font')
        matplotlib.rc('font', **font)

    fig = plt.figure()
    ax_main = fig.add_subplot(211)
    ax_res = fig.add_subplot(212)
    f_class = args.f()
    c_popt = None
    c_uncerts = None
    c_rchisq = None

    if args.c:
        data = unpack_data(args.c, args.delimiter)[0]
        (c_popt, c_uncerts, c_rchisq) = test_hyp(f_class, data)
        update_graphs(f_class, 'control', data, c_popt, fig, ax_main, ax_res)
        f_class.post_process(c_popt, c_uncerts, c_rchisq, ax_main, ax_res,
                             'control')
    for path in args.i:
        data = unpack_data(path, args.delimiter, filtr=True,
                           split_column=args.split)
        for i, (name, dat) in enumerate(data):
            (popt, uncerts, rchisq) = test_hyp(f_class, dat)
            update_graphs(f_class, name, dat, popt, fig, ax_main, ax_res)
            f_class.post_process(
                popt,
                uncerts,
                rchisq,
                ax_main,
                ax_res,
                name)
            f_class.notify(popt, uncerts, rchisq, name)
            if args.r:
                rpt = f_class.report(c_popt, popt, c_uncerts, uncerts, c_rchisq,
                                     rchisq, name)
                write_report(args.r, rpt)

    if args.s:
        fig.savefig(args.s)
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
