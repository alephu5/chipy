Congratulations on getting copy of ChiPy! This program will help you to compute the reduced chi-squared for your experimental data, and display it with minimal headache. There is no GUI and you must configure the program by writing python code, but the benefit of this approach is rapid development for me and a versatile environment for you.

Requirements:
Python version >=3.0
SciPy version >=0.17.0
numpy version >=1.10.4
matplotlib version >= 1.5.1

Quick start:
The experimental data should be stored in a '.csv' file with columns alternate between measurements and their errors. The last pair of columns are assumed to correspond to the dependent variable (a scalar), and all those preceeding are independent variables.

Optional Arguments:
  -h, --help            show this help message and exit
  -f f(X, a, b, ...)    A string of the form [module].[function] which
                        represents the theoretical model that you expect
                        between the data. If no model is supplied, a null
                        hypothesis is assumed
  -i [INPUT [INPUT ...]]
                        A list of paths to .csv files containing 
			experimental data.
  --delimiter DELIMITER
                        Character used to separate columns. Defaults to ','.
  -l Layout file        Path to python script containing matplotlib layout
                        parameters. Do not include .py extension.

Example usage:
There are some files included with this program to show you how to use it. Below is an example using every available argument.

python3 ./chipy.py -f sample_funcs.I -i sample_data.csv --delimiter=',' -l sample_layout

If you do not include a function then a constant is assumed: f(X) = c for all independent variables, meaning that the dependent variable is not affected by changes to the experimental parameters.

(Slightly more) Detailed instructions:
Hypothesis function:
The 'function' supplied with the '-f' argument is actually a class. This means that alongside your hypothesis, you can customise graphs and write custom messages after fitting a curve, and perform pre-processing on the data from the '.csv' file i.e applying a 'mod 360' operation to angles so that 721 deg becomes 1 deg. You can also have a set of post-processes on the reduced chi-square results, fitted parameters and graph. This is very useful for extracting important results and adding finishing touches to graphs.

If you don't know what classes are, don't worry. If you look at sample_funcs.py you will see that the class is basically a container for a set of functions and some variables, although every function within the class must have 'self' as the first argument for technical reasons that the internet will explain. The hypothesis function in the class must be called 'f' and be of the form
	f(X, a, b, ...) = Y
where X is the independent variable and (a, b, ...) are parameters to be optimised. In a polynomial these will just be the coefficients.

If you have very complicated procedures, you can write other functions outside the class i.e. to read a database or send an email, then call these from the class function.

Even if you don't want pre or post-processing you must leave these in place so that the program finds functions it expects.

Layout:
In the layout file you can adjust graph fonts.
