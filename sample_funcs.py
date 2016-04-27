#! /usr/bin/python3

"""Define functions here for use with ChiPy. Functions must be of the form
    f(X, params) = Y
   where X is an M by N array of measured independent variables, params are
   free variables to be optimised and Y is an N-length sequence of measured
   dependent variables data."""

import numpy as np
from chipy import Hypothesis, uncert


class I(Hypothesis):
    params = ['Peak voltage', 'Polarisation angle', 'Offset voltage']
    units = ['V', 'deg', 'V']
    bounds = (np.array([0, 0, 0]), np.array([10, 180, 1]))
    xlabel = 'Angle (deg)'
    ylabel = 'Voltage (V)'

    def f(self, ang, I0, polarisation, offset):
        """The hypothesis function. Compare measurements of voltage against
        f(ang, ...), where the remaining parameters are constants to be
        determined."""
        return I0 * np.cos(np.pi/180 * (ang - polarisation))**2 + offset

    def post_process(self, popt, uncerts, rchisq, data_name):
        """Provides a feature for modifying the results after they are obtained
        and modifying the graph to suit the specific form of data.

        We are fitting to a periodic function so there are an infinite number of
        solutions. We use post-processing to ensure that they all come from
        the same quadrant for meaningful comparison."""
        # Ensures that angles are within same quadrants.
        # popt.itemset(1, (popt[1] + 90) % 180 - 90)
        pass

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
        # symbol then sugar in last place, so we extract these differently
        # from the numbers.
        if '^' in data_name:
            cat = [data_name.split('^')[-1]]
            data_name = data_name[:data_name.index('^')]
        else:
            cat = []

        row = ''.join([c for c in data_name
                       if (c in ('-.') or str.isdigit(c))][:-1])
        sugar = data_name.split('-')[-1]
        row = row.split('-') + [sugar] + cat
        results = [0]*2*popt.shape[0]
        results[0::2] = list(c_popt - popt)
        results[1::2] = list(np.sqrt(1/2*(uncerts**2 + c_uncerts**2)))
        return row + list(map(str, results)) + [str(c_rchisq), str(rchisq)]

    def update_graph(self, data_name, data, popt, uncerts, color, fig,
                     ax_main, ax_res):
        ax_res.set_xlabel(self.xlabel)
        ax_res.set_ylabel('Measured - model (V)')
        ax_main.set_ylabel(self.ylabel)
        ax_main.vlines(x=popt[1], ymin=0, ymax=popt[0] + popt[2])

        (X, Xerr), (Y, Yerr) = data
        model = self.f(X, *popt)

        s = np.argsort(X)

        ax_main.errorbar(X[s], Y[s], Yerr[s], label='Measured ' + data_name,
                         lw=0, marker='x', ms=0.1, capsize=1, color='black')
        ax_main.plot(X[s], model[s], label='Model ' + data_name, lw=0.5,
                     color=color)
        sigma = uncert(self.f, X, Y, Xerr, Yerr, uncerts, popt)
        ax_res.errorbar(X[s], Y[s] - model[s], sigma[s], label=data_name,
                        lw=0.1, color=color)

        # ax_main.legend(fontsize=6)
        # ax_res.legend(fontsize=6)


class SpecRot(Hypothesis):
    params = ['Specific rotation']
    units = ['$\deg$ mL/ g dm']
    xlabel = 'Length × Concentration (dm g / mL)'
    ylabel = 'Angle (deg)'

    def f(self, LC, specrot):
        return specrot * LC

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
        # symbol then sugar in last place, so we extract these differently
        # from the numbers.
        if '^' in data_name:
            cat = [data_name.split('^')[-1]]
            data_name = data_name[:data_name.index('^')]
        else:
            cat = []

        row = ''.join([c for c in data_name
                       if (c in ('-.') or str.isdigit(c))][:-1])
        sugar = data_name.split('-')[-1]
        row = row.split('-') + [sugar] + cat
        results = [0]*2*popt.shape[0]
        results[0::2] = list(c_popt - popt)
        results[1::2] = list(np.sqrt((uncerts**2 + c_uncerts**2)))
        return row + list(map(str, results)) + [str(c_rchisq), str(rchisq)]

    def update_graph(self, data_name, data, popt, uncerts, color,
                     fig, ax_main, ax_res):
        ax_res.set_xlabel(self.xlabel)
        ax_res.set_ylabel('Measured - model (deg)')
        ax_main.set_ylabel(self.ylabel)

        (X, Xerr), (Y, Yerr) = data
        model = self.f(X, *popt)

        s = np.argsort(X)

        ax_main.errorbar(X[s], Y[s], Yerr[s], label=data_name,
                         marker='x', lw=0, ms=10, color=color)
        ax_main.plot(X[s], model[s], lw=2, color=color)
        sigma = uncert(self.f, X, Y, Xerr, Yerr, uncerts, popt)
        ax_res.errorbar(X[s], Y[s] - model[s], sigma[s], label=data_name,
                        lw=2, color=color)

        ax_res.legend()


class LinearReaction(Hypothesis):
    params = ['Gradient', 'Y-intercept']
    units = ['deg s$^-1$', 'deg']

    def f(self, X, c, m):
        (t_length, conc, acid_vol, delay, time) = X
        return m * time/60000 + c

    def report(self, c_popt, popt, c_uncerts, uncerts, c_rchisq, rchisq,
               data_name):
        if c_popt is None:
            c_popt = np.zeros(popt.shape)
            c_uncerts = np.zeros(uncerts.shape)
        # Naming convention has category column value in last place after a '^'
        # symbol then sugar in last place, so we extract these differently
        # from the numbers.
        if '^' in data_name:
            cat = [data_name.split('^')[-1]]
            data_name = data_name[:data_name.index('^')]
        else:
            cat = []

        row = ''.join([c for c in data_name
                       if (c in ('-.') or str.isdigit(c))][:-1])
        sugar = data_name.split('-')[-1]
        row = row.split('-') + [sugar] + cat
        results = [0]*2*popt.shape[0]
        results[0::2] = list(popt)
        results[1::2] = list(np.sqrt(1/2*(uncerts**2 + c_uncerts**2)))
        return row + list(map(str, results)) + [str(c_rchisq), str(rchisq)]

    def update_graph(self, data_name, data, popt, uncerts, color, fig,
                     ax_main, ax_res):
        ax_res.set_xlabel('Time (min)')
        ax_main.set_xlabel('Time (min)')
        ax_main.set_ylabel('Polarisation (deg)')
        ax_res.set_ylabel('Measured - model (deg)')

        (X, Xerr), (Y, Yerr) = data
        model = self.f(X, *popt)

        s = np.argsort(X[-1])
        C = -popt[-1] * X[-1]/60000

        ax_main.errorbar(X[-1, s]/60000, Y[s] + C[s], Yerr[s], label=data_name,
                         marker='x', lw=0, ms=0.1, color='black')
        ax_main.plot(X[-1, s]/60000, model[s] + C[s], lw=1, color=color)
        sigma = uncert(self.f, X, Y, Xerr, Yerr, uncerts, popt)
        ax_res.errorbar(X[-1, s]/60000, Y[s] - model[s], sigma[s],
                        label=data_name, lw=0.1, color=color)


class Mutarotation(Hypothesis):
    params = ['lambda', 'offset', 'coefficient', 'gradient']
    units = ['1/min', 'deg', 'g/L deg', 'deg/min']

    def f(self, time, lam, offset, coeff, gradient):
        return coeff * np.exp(-lam * time/60000) + offset + gradient*time/60000

    def report(self, c_popt, popt, c_uncerts, uncerts, c_rchisq, rchisq,
               data_name):
        if c_popt is None:
            c_popt = np.zeros(popt.shape)
            c_uncerts = np.zeros(uncerts.shape)
        # Naming convention has category column value in last place after a '^'
        # symbol then sugar in last place, so we extract these differently
        # from the numbers.
        if '^' in data_name:
            cat = [data_name.split('^')[-1]]
            data_name = data_name[:data_name.index('^')]
        else:
            cat = []

        row = ''.join([c for c in data_name
                       if (c in ('-.') or str.isdigit(c))][:-1])
        sugar = data_name.split('-')[-1]
        row = row.split('-') + [sugar] + cat
        results = [0]*2*popt.shape[0]
        results[0::2] = list(popt - c_popt)
        results[1::2] = list(np.sqrt(1/2*(uncerts**2 + c_uncerts**2)))
        return row + list(map(str, results)) + [str(c_rchisq), str(rchisq)]

    def update_graph(self, data_name, data, popt, uncerts, color, fig,
                     ax_main, ax_res):
        ax_res.set_xlabel('Time (min)')
        ax_main.set_ylabel('Polarisation (deg)')
        ax_res.set_ylabel('Measured - Model (deg)')

        (X, Xerr), (Y, Yerr) = data
        model = self.f(X, *popt)

        s = np.argsort(X)
        C = -popt[3] * X/60000

        ax_main.errorbar(X[s]/60000 + 6, Y[s] + C[s], Yerr[s], label=data_name,
                         marker='x', lw=0, ms=0.1, color='black')
        ax_main.plot(X[s]/60000 + 6, model[s] + C[s], lw=1, color=color)
        sigma = uncert(self.f, X + 6, Y, Xerr, Yerr, uncerts, popt)
        ax_res.errorbar(X[s]/60000 + 6, Y[s] - model[s], sigma[s],
                        label=data_name, lw=0.1, color=color)


class CorrExpReaction(Hypothesis):
    # Same as previous, but the data is automatically corrected for the linear
    # drift.
    params = ['lambda', 'a_s', 'a_F + a_G', 'gradient']
    units = ['1/m', 'deg ml / dm mol', 'deg ml / dm mol', 'deg/m']

    def f(self, X, lam, a_S, a_FplusG, grad):
        (t_length, conc, acid_vol, delay, time) = X
        LC = t_length * conc
        return (LC*((a_S - a_FplusG) * np.exp(-lam * (time + delay)/60000)
                + (a_FplusG)) + grad*time/60000)

    def report(self, c_popt, popt, c_uncerts, uncerts, c_rchisq, rchisq,
               data_name):
        if c_popt is None:
            c_popt = np.zeros(popt.shape)
            c_uncerts = np.zeros(uncerts.shape)
        # Naming convention has category column value in last place after a '^'
        # symbol then sugar in last place, so we extract these differently
        # from the numbers.
        if '^' in data_name:
            cat = [data_name.split('^')[-1]]
            data_name = data_name[:data_name.index('^')]
        else:
            cat = []

        row = ''.join([c for c in data_name
                       if (c in ('-.') or str.isdigit(c))][:-1])
        sugar = data_name.split('-')[-1]
        row = row.split('-') + [sugar] + cat
        results = [0]*2*popt.shape[0]
        results[0::2] = list(popt - c_popt)
        results[1::2] = list(np.sqrt(1/2*(uncerts**2 + c_uncerts**2)))
        return row + list(map(str, results)) + [str(c_rchisq), str(rchisq)]

    def update_graph(self, data_name, data, popt, uncerts, color, fig,
                     ax_main, ax_res):
        ax_res.set_xlabel('Time (min)')
        ax_main.set_ylabel('Polarisation (deg)')
        ax_res.set_ylabel('Measured - model (deg)')

        (X, Xerr), (Y, Yerr) = data
        model = self.f(X, *popt)

        s = np.argsort(X[-1])
        C = -popt[-1] * X[-1]/60000

        time = (X[-1, s] + X[-2])/60000
        ax_main.errorbar(time, Y[s] + C[s], Yerr[s], label=data_name,
                         marker='x', lw=0, ms=0.1, color='black')
        ax_main.plot(time, model[s] + C[s], lw=1, color=color)
        sigma = uncert(self.f, X, Y, Xerr, Yerr, uncerts, popt)
        ax_res.errorbar(time, Y[s] - model[s], sigma[s],
                        label=data_name, lw=0.1, color=color)


class MultiRot(Hypothesis):
    # A more complex model that accounts for the mutarotation of fructose nad
    # glucose.
    params = ['lam',   'k_f', 'k_g', 'a_f0', 'a_flim', 'a_g0', 'a_glim', 'a_s']
    units  = ['1/min', 'x',   'x',   'x',    'x',      'x',    'x',      'x']

    p0 = np.array([0.05055, 0.1046, 0.88, -19871, -13457, 16844, 7908, 19579])
    bounds = (p0 -
         np.array([0.04, 0.07, 0.07, 1000, 2000, 1000, 1000, 1000]),
              p0 +
         np.array([0.04, 0.07, 0.07, 1000, 2000, 1000, 1000, 1000]))

    def f(self, X, lam, k_f, k_g, a_f0, a_flim, a_g0, a_glim, a_s):
        (t_length, conc, acid_vol, delay, time) = X
        LC = t_length * conc

        #a_f0 = -28848
        #a_flim = -9134.8
        #a_g0 = 8164
        #a_glim = 2905
        #a_s = 19579.3598
        #a_f0 = -19871
        #a_flim = -13457
        #a_g0 = 16844
        #a_glim = 7908
        #a_s = 19579

        kaa = (k_f, a_f0, a_flim), (k_g, a_g0, a_glim)
        rotation = 0
        elt = np.exp(-lam*(time + delay)/60000)
        for k, a0, ainf in kaa:
            ekt = np.exp(-k*(time + delay)/60000)
            A = lam * (a0 - ainf) * (elt - ekt) / (k - lam)
            B = ainf * (1 - elt)
            rotation += A + B

        return LC * (a_s*elt + rotation)

    def report(self, c_popt, popt, c_uncerts, uncerts, c_rchisq, rchisq,
               data_name):
        if c_popt is None:
            c_popt = np.zeros(popt.shape)
            c_uncerts = np.zeros(uncerts.shape)
        # Naming convention has category column value in last place after a '^'
        # symbol then sugar in last place, so we extract these differently
        # from the numbers.
        if '^' in data_name:
            cat = [data_name.split('^')[-1]]
            data_name = data_name[:data_name.index('^')]
        else:
            cat = []

        row = ''.join([c for c in data_name
                       if (c in ('-.') or str.isdigit(c))][:-1])
        sugar = data_name.split('-')[-1]
        row = row.split('-') + [sugar] + cat
        results = [0]*2*popt.shape[0]
        results[0::2] = list(popt - c_popt)
        results[1::2] = list(np.sqrt(1/2*(uncerts**2 + c_uncerts**2)))
        return row + list(map(str, results)) + [str(c_rchisq), str(rchisq)]

    def update_graph(self, data_name, data, popt, uncerts, color, fig,
                     ax_main, ax_res):
        ax_res.set_xlabel('Time (min)')
        ax_main.set_ylabel('Polarisation (deg)')
        ax_res.set_ylabel('Measured - model (deg)')
        (X, Xerr), (Y, Yerr) = data
        model = self.f(X, *popt)

        s = np.argsort(X[-1])
        C = 0 * X[-1]/60000

        ax_main.errorbar((X[-1, s] + X[-2])/60000, Y[s] + C[s], Yerr[s], label=data_name,
                         marker='x', lw=0, ms=0.1, color='black')
        ax_main.plot((X[-1, s] + X[-2])/60000, model[s] - C[s], lw=1, color=color)
        # sigma = uncert(self.f, X, Y, Xerr, Yerr, uncerts, popt)
        ax_res.plot((X[-1, s] + X[-2])/60000, Y[s] - model[s],
                    label=data_name, lw=0.3, color=color)
