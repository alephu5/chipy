#!/usr/bin/python3

import argparse
import numpy as np
import os
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


class Hypothesis:

    """Base class used to build hypothesis classes. Provides basic forms of the
    necessary functions, to avoid boilerplate in the plugins."""
    # params = ['Gradient', 'Offset'] # Set meaningful names for free
    #                                   parameters to display to the user.
    # units = ['m s$^-1$', 'm']

    bounds = (-np.inf, np.inf)  # Limits on the fitted parameters

    def f(self, X, a, b):
        """The hypothesis function; which relates independent variable(s) X
        to dependent variable Y as f(X, a, b,...) = Y. The remaining parameters
        are constants that can be adjusted to fit the data.

        If there are multiple independent variables X should be array-like."""
        return a*X + b

    def pre_process(self, X):
        """Gives the option of modifying the function before using it to fit
        the hypothesis. This feature was added to use the modulus 180 of an
        angle (ie 270deg == 90 deg)."""
        return X

    def post_process(self, popt, uncerts, rchisq, data_name):
        """Provides a feature for modifying the results after they are obtained.
        """

        pass

    def update_graph(self, data_name, data, popt, fig, ax_main, ax_res):
        ax_main.set_xlabel(self.params[0] + '(' + self.units[0] + ')')
        ax_main.set_ylabel(self.params[1] + '(' + self.units[1] + ')')
        ax_main.vlines(x=popt[1], ymin=0, ymax=popt[0] + popt[2])

        ax_res.set_title('Residuals')

        (X, Xerr), (Y, Yerr) = data
        S = np.sort(X)
        model = self.f(S, *popt)

        ax_main.errorbar(
            X, Y, Yerr, label='Measured ' + data_name, lw=0, marker='x',
            ms=0.1, mew=2, capsize=1)
        ax_main.plot(S, model, label='Model ' + data_name, lw=2)
        ax_res.errorbar(X, Y - self.f(X, *popt), np.sqrt(Xerr**2 + Yerr**2),
                        label=data_name, lw=0.1)
        ax_res.legend()
        ax_main.legend()

    def notify(self, popt, uncerts, rchisq, data_name):
        """Provides a function for communicating to the user at run-time about
        the results."""
        print(data_name + ':')
        for i, param in enumerate(self.params):
            uncert = '%s' % float('%.1g' % uncerts[i])
            precision = len(uncert)
            if '.' in uncert:
                precision -= uncert.index('.')
            print(param, '=', round(popt[i], precision), '+/-',
                  uncert, self.units[i])
        print('Having reduced chi-squared', rchisq, os.linesep)

    def report(self, c_popt, popt, c_uncerts, uncerts, c_rchisq, rchisq,
               data_name):
        """Produces a report about the fit that can be written to a csv file
        for further processing. This is not displayed on the screen and will
        only be used if the user requests a report and specifies a write-path
        at runtime in chipy."""
        if c_popt is None:
            c_popt = np.zeros(popt.shape)
            c_uncerts = np.zeros(uncerts.shape)
        # Naming convention has category column value in last place after a '^'
        if '^' in data_name:
            cat = [data_name.split('^')[-1]]
            data_name = data_name[:data_name.index('^')]
        else:
            cat = []

        row = ''.join([c for c in data_name if (c in ('-.') or str.isdigit(c))])
        row = row.split('-') + cat
        results = [0]*2*popt.shape[0]
        results[0::2] = list(c_popt - popt)
        results[1::2] = list(np.sqrt(1/2*(uncerts**2 + c_uncerts**2)))
        return row + list(map(str, results)) + [str(c_rchisq), str(rchisq)]


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
    parser.add_argument('--filter', dest='filtr', action='store_true',
                        help="""Remove values from dataset with errors larger
                        than their magnitude.""")
    parser.add_argument('--no-filter', dest='filtr', action='store_true',
                        help="""Use all data available.""")
    parser.set_defaults(show=True)
    parser.set_defaults(filtr=False)
    return parser.parse_args()


def uncert(f, X, Y, Xerr, Yerr, *fparams):
    try:
        # To compute the gradient, X must be presented as a set of arrays. If
        # it contains just scalar values, then these must be converted to the
        # necessary format using reshape.

        X.shape[1]
        X_space = X
    except IndexError:
        X_space = X.reshape((1, X.shape[0]))

    grad = np.array([scipy.optimize.approx_fprime(x, f, 1e-8, *fparams)
                     for x in X_space.transpose()]).transpose()
    return np.sqrt(np.sum((grad*Xerr)**2, axis=0) + Yerr**2)


def chisq(f, X, Y, Xerr, Yerr, *fparams):
    sigma = uncert(f, X, Y, Xerr, Yerr, *fparams)
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


def min_chisq(f, X, Y, Xerr, Yerr, bounds, p0=None):
    popt, pcov = scipy.optimize.curve_fit(f, X, Y, sigma=Yerr,
                                          p0=p0, bounds=bounds)
    dof = len(Y) - len(popt)
    rchisq = chisq(f, X, Y, Xerr, Yerr, *popt) / dof
    # rchisq_err = chisq_err(f, X, Y, Yerr, popt) / dof
    # Removed because of questionable accuracy for multiple parameters.
    uncerts = np.sqrt(pcov.diagonal())
    return popt, uncerts, rchisq  # , rchisq_err


def test_hyp(f_class, data, cpopt=None):
    f = f_class.f
    (X, Xerr), (Y, Yerr) = data
    X = f_class.pre_process(X)
    return min_chisq(f, X, Y, Xerr, Yerr, f_class.bounds, cpopt)


def analysis_update(data, f_class, color, graph, report,
                    control=(None, None, None), notify=True):
    c_popt, c_uncerts, c_rchisq = control
    if graph:
        fig, axes = graph

    for i, (name, dat) in enumerate(data):
        (popt, uncerts, rchisq) = test_hyp(f_class, dat)
        f_class.post_process(popt, uncerts, rchisq, name)
        f_class.update_graph(name, dat, popt, color, fig, *axes)

        if notify:
            f_class.notify(popt, uncerts, rchisq, name)

        if report:
            rpt = f_class.report(
                c_popt,
                popt,
                c_uncerts,
                uncerts,
                c_rchisq,
                rchisq,
                name)
            write_report(report, rpt)

    return popt, uncerts, rchisq


def main():
    args = parseargs()
    if args.l:
        # Load layout
        font = load_attribute(args.l + '.font')
        matplotlib.rc('font', **font)

    graph = None
    if args.show or args.s:
        fig = plt.figure()
        ax_main = fig.add_subplot(211, label='Main')
        ax_res = fig.add_subplot(212, label='Residuals')
        graph = (fig, (ax_main, ax_res))

    f_class = args.f()
    control = (None, None, None)

    if args.c:
        _, dat = unpack_data(args.c, args.delimiter).__next__()
        data = [('control', dat)]
        control = analysis_update(data, f_class, 'blue', graph, report=False,
                                  control=control, notify=False)

    for path in args.i:
        try:
            data = unpack_data(path, args.delimiter, filtr=args.filtr,
                               split_column=args.split)
            analysis_update(data, f_class, 'green', graph, args.r,
                            control=control, notify=True)

        except Exception as e:
            print ('Could not process', path, os.linesep)
            raise e

    if args.s:
        bname = os.path.basename(path)
        name = os.path.splitext(bname)[0]
        fig.savefig(os.path.join(args.s, name + '.png'), dpi=160,
                    bbox_inches='tight', pad_inches=0)

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
