# Z0
 
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

There are two files in this directory.
1.18_Grope-Data.ipynb which is the analysis of the grope data.
2. z0_experiment_group_18 which is the analysis of the further exercise i.e, simulation of mc data and analysis of detector data.

In both the files you dont need anything else to run the code, once you activate the environment.
You need to run all the cells one by one and if it stops in between you need to restart the kernel again.

Also in the detector data we have used 
daten_5.root
daten_5.lum
