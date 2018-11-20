#!/usr/bin/python3

import argparse
import numpy as np
import os
import sys
from copy import deepcopy
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import Grid
from fileops import load_attribute, unpack_data, write_report
from decimal import Decimal

DESCRIPTION = """Automates common operations for chi-squared analysis used in
level 2 labs. The data should be presented as columns alternating between
measurements and their errors. The last pair of columns are assumed to
correspond to the dependent variable (a scalar), and those preceeding are an
independent vector."""


class Hypothesis:

    """Base class used to build hypothesis classes. Provides basic forms of the
    necessary functions, to avoid boilerplate in the plugins."""
    # Set meaningful names for free parameters to display to the user.
    params = ['Constant']
    units = ['']
    xlabel = 'x'
    ylabel = 'y'
    bounds = (-np.inf, np.inf)  # Limits on the fitted parameters
    p0 = None

    def f(self, X, c):
        """The hypothesis function; which relates independent variable(s) X
        to dependent variable Y as f(X, a, b,...) = Y. The remaining parameters
        are constants that can be adjusted to fit the data.

        If there are multiple independent variables X should be array-like."""
        return np.repeat(c, np.size(X))

    def pre_process(self, A, B):
        """Gives the option of modifying the function before using it to fit
        the hypothesis. This feature was added to use the modulus 180 of an
        angle (ie 270deg == 90 deg)."""
        return A, B

    def post_process(self, popt, uncerts, rchisq, data_name):
        """Provides a feature for modifying the results after they are obtained.
        """

        pass

    def update_graph(self, data_name, data, popt, Perr, color, fig,
                     ax_main, ax_res):
        ax_res.set_xlabel(self.xlabel)
        ax_res.set_ylabel('Measured - model')
        ax_main.set_ylabel(self.ylabel)

        (X, Xerr), (Y, Yerr) = data
        model = self.f(X, *popt)

        s = np.argsort(X)

        ax_main.errorbar(X[s], Y[s], Yerr[s], label='Measured ' + data_name,
                         linestyle='None', marker='x', ms=10, capsize=0,
                         color='black')
        ax_main.plot(X[s], model[s], label='Model ' + data_name, lw=1,
                     color=color)

        sigma = np.sqrt(uncert(self.f, X, Y, Xerr, Perr, popt)**2 + Yerr**2)
        ax_res.plot(X[s], np.zeros(len(X)), linestyle='--', color='black',
                    lw=0.5)
        ax_res.errorbar(X[s], Y[s] - model[s], sigma[s], label=data_name,
                        linestyle='None', marker='x', ms=10, capsize=0,
                        color=color)

    def notify(self, popt, uncerts, rchisq, data_name):
        """Provides a function for communicating to the user at run-time about
        the results."""
        print(data_name + ':')
        for i, param in enumerate(self.params):
            prec = 0
            if uncerts[i] != 0:
                prec = int(np.log10(uncerts[i]))
            uncert = uncerts[i]
            try:
                uncert = Decimal(uncerts[i]).quantize(Decimal(10)
                                                      ** (prec - 1)).normalize()
            except:
                pass
            val = popt[i]
            try:
                val = Decimal(popt[i]).quantize(Decimal(10)
                                                ** (prec - 1)).normalize()
            except:
                # Don't try rounding value.
                pass

            print(param, '=', val, '+/-', uncert, self.units[i])
        print('Having reduced chi-squared', rchisq)
        print()

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
        default=Hypothesis, metavar='f(X)',
        help="""A string of the form [module].[class] which
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


def uncert(f, X, Y, Xerr, Perr, *fparams):
    if X.ndim <= 1:
        X_space = X.reshape((X.shape[0], 1))
        Xerr_space = Xerr.reshape(Xerr.shape[0], 1)
    else:
        X_space = X.transpose()
        Xerr_space = Xerr.transpose()

    P_space = np.array(fparams, ndmin=2)
    if Perr.ndim <= 1:
        Perr_space = Perr.reshape(1, Perr.shape[0])
    else:
        Perr_space = Perr.tranpose()

    idx = np.repeat(X_space.shape[1], len(fparams))
    T = np.insert(X_space, idx, P_space, axis=1)
    Terr = np.insert(Xerr_space, idx, Perr_space, axis=1)

    def g(t):
        x_length = X_space.shape[1]
        X = t[:x_length]
        args = t[x_length:]
        return f(X, *args)

    grad = np.array([scipy.optimize.approx_fprime(t, g, 1e-8)
                     for t in T])
    grad[np.isinf(grad)] = 0
    grad[np.isnan(grad)] = 0
    Terr[np.isinf(Terr)] = 0
    Terr[np.isnan(Terr)] = 0
    return np.sqrt(np.sum((grad*Terr)**2, axis=1))


def uncert_bad(f, X, Y, Xerr, Perr, *fparams):
    return 0


def chisq(f, X, Y, Xerr, Yerr, Perr, *fparams):
    sigma = np.sqrt(uncert(f, X, Y, Xerr, Perr, *fparams)**2 + Yerr**2)
    chis = ((f(X, *fparams) - Y) / sigma) ** 2
    chis[np.isnan(chis)] = 0
    chis[np.isinf(chis)] = 0
    return chis.sum()


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
    popt, pcov = scipy.optimize.curve_fit(f, X, Y, sigma=Yerr, p0=p0,
                                          bounds=bounds, absolute_sigma=True)
    dof = len(Y) - len(popt)
    Perr = np.sqrt(pcov.diagonal())
    rchisq = chisq(f, X, Y, Xerr, Yerr, Perr, *popt) / dof
    # rchisq_err = chisq_err(f, X, Y, Yerr, popt) / dof
    # Removed because of questionable accuracy for multiple parameters.
    return popt, Perr, rchisq  # , rchisq_err


def test_hyp(f_class, data, cpopt=None):
    f = f_class.f
    (X, Xerr), (Y, Yerr) = data
    if f_class.p0 is not None:
        p0 = f_class.p0
    else:
        p0 = cpopt
    return min_chisq(f, X, Y, Xerr, Yerr, f_class.bounds, p0)


def analysis_update(data, f_class, color, graph, report,
                    control=(None, None, None), notify=True):
    c_popt, c_uncerts, c_rchisq = control
    if graph:
        fig, axes = graph

    for i, (name, dat) in enumerate(data):
        dat = f_class.pre_process(*dat)
        (popt, uncerts, rchisq) = test_hyp(f_class, dat)
        f_class.post_process(popt, uncerts, rchisq, name)
        if graph:
            f_class.update_graph(name, dat, popt, uncerts, color, fig, *axes)

        if notify:
            f_class.notify(popt, uncerts, rchisq, name)

        if report:
            rpt = f_class.report(c_popt, popt, c_uncerts, uncerts, c_rchisq,
                                 rchisq, name)
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
        ax_main, ax_res = Grid(fig, rect=111, nrows_ncols=(2, 1),
                               axes_pad=0.2, label_mode='L')
        # ax_main.get_xaxis().set_visible(False)

        # div = make_axes_locatable(ax_main)
        # ax_res = div.append_axes('bottom', pad=0.05, label='Residuals',
        #                          size='50%')
        # ax_res = fig.add_subplot(212, label='Residuals')
        graph = (fig, (ax_main, ax_res))

    f_class = args.f()
    control = (None, None, None)

    colors = iter(cm.brg(np.linspace(0, 1, len(args.i) + bool(args.c))))

    if args.c:
        _, dat = unpack_data(args.c, args.delimiter).__next__()
        data = [('control', dat)]
        control = analysis_update(data, f_class, colors.__next__(), graph,
                                  report=False, control=control, notify=False)
    for path in args.i:
        color = colors.__next__()
        try:
            data = unpack_data(path, args.delimiter, filtr=args.filtr,
                               split_column=args.split)
            analysis_update(data, f_class, color, graph, args.r,
                            control=control, notify=True)

        except Exception as e:
            raise e
            print ('Could not process', path, os.linesep)
            return None

    if args.s:
        plt.tight_layout()
        bname = os.path.basename(path)
        name = os.path.splitext(bname)[0]
        fig.savefig(os.path.join(args.s, name + '.png'), dpi=160,
                    bbox_inches='tight', pad_inches=0)

    if args.show:
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
