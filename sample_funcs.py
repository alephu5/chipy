#! /usr/bin/python3

"""Define functions here for use with ChiPy. Functions must be of the form
    f(X, params) = Y
   where X is an M by N array of measured independent variables, params are
   free variables to be optimised and Y is an N-length sequence of measured
   dependent variables data."""

from numpy import cos, pi, sqrt


class I:
    params = ['Peak voltage', 'Polarisation angle', 'Offset voltage']
    units = ['V', 'deg', 'V']

    def f(self, ang, I0, polarisation, offset):
        return I0 * cos(pi/180 * (ang - polarisation))**2 + offset

    def pre_process(self, ang):
        return ang

    def post_process(self, popt, pcov, rchisq, ax, data_name):
        # Ensures that angles are within same quadrants.
        popt.itemset(1, popt[1] % 180)
        print(data_name + ':')
        uncerts = sqrt(pcov.diagonal())
        for i, param in enumerate(self.params):
            uncert = '%s' % float('%.1g' % uncerts[i])
            precision = len(uncert) - uncert.index('.')
            print(param, '=', round(popt[i], precision), '+/-',
                  uncert, self.units[i])
        print('Having reduced chi-squared', rchisq)
        ax.set_xlabel(self.params[0] + '(' + self.units[0] + ')')
        ax.set_ylabel(self.params[1] + '(' + self.units[1] + ')')
        ax.vlines(x=popt[1], ymin=0, ymax=popt[0] + popt[2])
        print()


class Drude:
    params = ['A', '$\lambda_0$']
    units = ['rad m$^4$/g', 'nm']

    def f(self, wavelength, A, lambda_0):
        return A / (wavelength ^ 2 - lambda_0 ^ 2)

    def post_process(self, popt, pcov, rchisq, ax, data_name):
        print(data_name + ':')
        uncerts = sqrt(pcov.diagonal())
        for i, param in enumerate(self.params):
            uncert = '%s' % float('%.1g' % uncerts[i])
            precision = len(uncert) - uncert.index('.')
            print(param, '=', round(popt[i], precision), '+/-',
                  uncert, self.units[i])
        print('Having reduced chi-squared', rchisq)
        ax.set_xlabel(self.params[0] + '(' + self.units[0] + ')')
        ax.set_ylabel(self.params[1] + '(' + self.units[1] + ')')
        print()
