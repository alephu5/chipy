#! /usr/bin/python3

"""Define functions here for use with ChiPy. Functions must be of the form
    f(X, params) = Y
   where X is an M by N array of measured independent variables, params are
   free variables to be optimised and Y is an N-length sequence of measured
   dependent variables data."""

from numpy import cos, pi


def I(ang, I0, offset):
    return I0 * cos(pi * ang/180 - offset)**2
