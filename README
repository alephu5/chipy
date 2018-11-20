#ChiPy - Chi² Hypothesis testing

## Preamble
This is a small command-line program that can facilitate hypothesis testing by computing the reduced chi-squared. I built it during my undergraduate physics degree and so no longer update it. The GitHub repository is still active so if you have a bug-fix or want to implement a new feature please submit a pull request.

Requirements:
Python version >=3.0
SciPy version >=0.17.0
numpy version >=1.10.4
matplotlib version >= 1.5.1

## Introduction
The experimental data should be stored in a '.csv' file with columns alternate between measurements and their errors (xᵢ, σᵢ; y, Σ). The last pair of columns are assumed to correspond to the dependent variable (a scalar), and all those preceeding are independent variables. A mathematical function f with adjustable parameters is fitted to the data f(x₁, ..., xₙ) = y. The resultant best-fit model is graphed, the optimised parameter values are listed with their respective errors, and the reduced chi-squared statistic is computed.

## Quick start
The most basic functionality can be seen by running:

`python3 ./chipy.py -i demo/sample_data.csv`

which will try to fit a horizontal line f(x₁, ..., xₙ) = y = c to the data.

If you run the previous command you will see that the model does not describe the data at all, so let's try some different ones. In the demo directory there is a python file called 'sample_models.py' that contains a linear and quadratic model, contained within the classes 'Linear' and 'Quadratic' respectively. We can fit these by running the commands:

`python3 ./chipy.py -i demo/sample_data.csv -f demo/sample_models.Linear`
`python3 ./chipy.py -i demo/sample_data.csv -f demo/sample_models.Quadratic`

## Detailed instructions
Optional arguments:
  -h, --help            show this help message and exit
  -f f(X)               A string of the form [module].[class] which represents
                        the theoretical model that you expect between the
                        data. If no model is supplied, a null hypothesis is
                        assumed
  -i [INPUT [INPUT ...]]
                        Columns of experimental data, assumed to be
                        alternating between experiment and errors.
  -c CONTROL            Uses the argument as a control condition. All fitted
                        parameters are subtracted from this when reporting the
                        final result and the errors are propagated.
  --delimiter DELIMITER
                        Character used to separate columns. Defaults to ','.
  -l LAYOUT FILE        Path to python script containing matplotlib layout
                        parameters.
  -s SAVE GRAPH         Automatically save graph in the specified directory.
  -r REPORT             Writes results to the file specified, using procedures
                        defined in the function file.
  --show                Show a graph.
  --no-show             Don't show a graph.
  --split SPLIT         Splits the file into seperate pieces; grouping them by
                        values in the column specified. The first column is 0.
  --filter              Remove values from dataset with errors larger than
                        their magnitude.
  --no-filter           Use all data available.

Hypothesis function:
The 'function' supplied with the '-f' argument is actually a class. This means that alongside your hypothesis, you can customise graphs and write custom messages after fitting a curve, and perform pre-processing on the data from the '.csv' file i.e applying a 'mod 360' operation to angles so that 721 deg becomes 1 deg. You can also have a set of post-processes on the reduced chi-square results, fitted parameters and graph. This is very useful for extracting important results and adding finishing touches to graphs.

If you don't know what classes are, don't worry. If you look at sample_funcs.py you will see that the class is basically a container for a set of functions and some variables, although every function within the class must have 'self' as the first argument for technical reasons that the internet will explain. The hypothesis function in the class must be called 'f' and be of the form
	f(X, a, b, ...) = Y
where X is the independent variable and (a, b, ...) are parameters to be optimised. In a polynomial these will just be the coefficients.

If you have very complicated procedures, you can write other functions outside the class i.e. to read a database or send an email, then call these from the class function.

Even if you don't want pre or post-processing you must leave these in place so that the program finds functions it expects.

Layout:
In the layout file you can adjust graph fonts.
