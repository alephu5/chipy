#! /usr/bin/python3

"""Define functions here for use with ChiPy. Functions must be of the form
    f(X, params) = Y
   where X is an M by N array of measured independent variables, params are
   free variables to be optimised and Y is an N-length sequence of measured
   dependent variables data."""

from numpy import cos, pi, sqrt, zeros


class I:
    params = ['Peak voltage', 'Polarisation angle', 'Offset voltage']
    units = ['V', 'deg', 'V']

    def f(self, ang, I0, polarisation, offset):
        return I0 * cos(pi/180 * (ang - polarisation))**2 + offset

    def pre_process(self, ang):
        return ang

    def post_process(self, popt, uncerts, rchisq, ax_main, ax_res,
                     data_name, *args):
        # Ensures that angles are within same quadrants.
        popt.itemset(1, popt[1] % 180)
        print(data_name + ':')
        for i, param in enumerate(self.params):
            uncert = '%s' % float('%.1g' % uncerts[i])
            precision = len(uncert) - uncert.index('.')
            print(param, '=', round(popt[i], precision), '+/-',
                  uncert, self.units[i])
        print('Having reduced chi-squared', rchisq)

        ax_main.set_xlabel(self.params[0] + '(' + self.units[0] + ')')
        ax_main.set_ylabel(self.params[1] + '(' + self.units[1] + ')')
        ax_main.vlines(x=popt[1], ymin=0, ymax=popt[0] + popt[2])
        print()

    def report(self, c_popt, popt, c_uncerts, uncerts, c_rchisq, rchisq,
               data_name):
        if not c_popt:
            c_popt = zeros(popt.shape)
            c_uncerts = zeros(uncerts.shape)
        if '^' in data_name:
            cat = data_name.split('^')[-1]
            data_name = data_name.split('^')[:-1]
        else:
            cat = ''

        row = data_name.split('-')
        for i, c in enumerate(row):
            fil = filter(str.isdigit, c)
            c = ''.join(list(fil))
            row[i] = c
        results = [0]*2*popt.shape[0]
        results[0::2] = list(popt - c_popt)
        results[1::2] = list(sqrt(1/2*(uncerts**2 + c_uncerts**2)))
        return row + list(map(str, results))


class Drude:
    params = ['A', '$\lambda_0$']
    units = ['rad m$^4$/g', 'nm']

    def f(self, wavelength, A, lambda_0):
        return A / (wavelength ^ 2 - lambda_0 ^ 2)

    def post_process(self, popt, uncerts, rchisq, ax, data_name):
        print(data_name + ':')
        for i, param in enumerate(self.params):
            uncert = '%s' % float('%.1g' % uncerts[i])
            precision = len(uncert) - uncert.index('.')
            print(param, '=', round(popt[i], precision), '+/-',
                  uncert, self.units[i])
        print('Having reduced chi-squared', rchisq)
        ax.set_xlabel(self.params[0] + '(' + self.units[0] + ')')
        ax.set_ylabel(self.params[1] + '(' + self.units[1] + ')')
        print()
