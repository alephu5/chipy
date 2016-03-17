#! /usr/bin/python3

from importlib import import_module
import numpy as np
import sys
import os

# Adds working directing to path variable, so modules can be imported.
sys.path.append(os.getcwd())


def unpack_data(path, delimiter, filtr=False, split_column=-1):
    """Measurements and errors are assumed to be alternating. The last
    pair of columns corresponds to the dependent variable
    while the preceeding are independent.

    If filtr is True, values larger than the error are removed.

    If split_column is given, the data is split into lumps with a column
    value in that column."""
    raw = np.loadtxt(path, delimiter=delimiter, skiprows=1)
    data_name = os.path.splitext(os.path.basename(path))[0]

    if split_column != -1:
        raws = split_file(raw, split_column, data_name)
    else:
        # Needed to generalise following iterative step.
        raws = [(data_name, raw)]
    for (name, raw) in raws:
        meas = raw[:, ::2].transpose()
        err = raw[:, 1::2].transpose()
        if filtr:
            test = (abs(meas) >= err).prod(axis=0)
            meas = np.compress(test, meas, axis=1)
            err = np.compress(test, err, axis=1)

        if meas.shape[0] == 2:
            A = (meas[:-1].ravel(), err[:-1].ravel())
            yield name, (A, (meas[-1], err[-1]))
        else:
            yield name, ((meas[:-1], err[:-1]), (meas[-1], err[-1]))


def split_file(data, split_column, data_name):
    """Opens the file specified by the path name and splits it according to
    groups categorised by the value in the specified column. Indexed with 0
    0 being the first"""
    keys = set(data[:, split_column])
    for key in keys:
        indices = (data[:, split_column] == key)
        datum = data[indices]
        yield data_name + '-' + str(key), np.delete(datum, split_column, 1)


def get_headings(path, delimiter):
    with open(path) as f:
        headings = f.readline()
        headings = headings.split(delimiter)
    return headings


def load_attribute(astring):
    m, a = astring.split('.')
    mdl = import_module(m)
    return getattr(mdl, a)


def write_report(report_path, result):
    with open(report_path, 'a') as f:
        f.write(','.join(result) + os.linesep)
