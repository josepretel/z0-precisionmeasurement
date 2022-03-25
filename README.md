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

# Instructions for runnig the files from group 8:

Please see the final versions (i.e. submission) in the following two jupyter notebook:
1. 01_z0_grope_tu_fs.ipynb for the GROPE data anlysis
2. 02_z0_detector_statistical_analysis_tu_fs.ipynb for the remaining analysis of the Monte Carlo and Detector data.

Details concerning the procedure of the analysis are briefly discussed in the relevant sections of the notebooks. The elaborate discussion of the theory and formulae will follow in the formal lab report.

Over the course of the second notebook some variables and dictionaries are adapted. In case errors or inconsistencies occur, we advise the user to restart the kernel of the notebook and run all cells consecutively.
