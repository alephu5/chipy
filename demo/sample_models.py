#! /usr/bin/python3

from chipy import Hypothesis


class Linear(Hypothesis):
    params = ['Gradient', 'Constant']
    units = ['mm/g', 'mm']
    bounds = ([-1, -2], [2, 1])
    xlabel = 'Mass (g)'
    ylabel = 'Length (mm)'

    def f(self, mass, grad, const):
        return mass * grad + const


class Quadratic(Hypothesis):
    params = ['Curvature']
    units = ['mm/g^2']
    bounds = (-1, 1)
    xlabel = 'Mass (g)'
    ylabel = 'Length (mm)'

    def f(self, weight, curv):
        return weight ** 2 * curv
