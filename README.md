# z0-precisionmeasurement

Z0 decay experiment for the FP2 laboratory course at the Albert-Ludwigs University of Freiburg.
The goal of this experiment is to carry out precision measurements and tests of the Standard Model using at LEP recorded with the OPAL detector

# Installing with conda

1. Install [anaconda](https://docs.anaconda.com/anaconda/install/).
2. `conda env create` reads the `environment.yml` file in this
   repository, creates a new env and installs all necessary packages
   into it.
3. Activate the new env: `conda activate z0-env`
4. Start `jupyter-lab` (or `jupyter-notebook` if you prefer) in the
   environment. This will usually open your browesr automatically. If
   not the link is printed to the console too.

# How to run the code

In order to reproduce our results you just have to run the code from top to bottom, i.e. in the order in which the single cells are given.
Since some variables are used several times, in order for them not to be overwritten you should strictly follow this order.
Thus, if something in the code does not work you might have to restart the kernel and run all.

In case that the calculation of the error on the inverse matrix does not work you might have to change the ranges or add parameters to the p0 list we added.

There are no extra packages that you have to use other than the ones you find in the first / second cell of the code.
The code file is called `z0_experiment_group_1.ipynb`.
The files needed in order to run the code are:
`daten_2.csv`,
`daten_2.root`,
`ee.root`,
`mm.root`,
`qq.root`,
`tt.root`.
