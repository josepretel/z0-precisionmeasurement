# FP2 - Z0-Experiment

## File: Grope_analysis_notebook 
* Start z0-env python environment and open jupyter notebook from there
* Import all package with first cell
* Execute all other cells from top to bottom

For orientation, please view the headings in markdown cells:
* 'Histograms': The histograms of the GROPE Data
* 'particle ID': The particle identification functions
* 'Program to test ID': Calculate effiencies of selection progress.

## File: analysis_Monte_Carlo_data
* Start z0-env python environment and open jupyter notebook from there
* Import all package with first cell

Headings (and where to find what):
* 'Data Import': The MC data is imported from the folder 'mc_z0experiments_main' from the same directory. The experimental data is impoerted from the folder 'data_z0experiment-main' again in the same directory.
* 'Histograms MC data': Contains the histograms of the MC data
* 'Histograms OPAL data': Contains the histograms of the experimental data from OPAL
* 'ParticleID': Contains the particle ID functions.
* 'Effiency Matrix': Here, the effiency matrix is calculated. 
* 'Errors': The error Matrix is calculated.
* 'Exercise 2': Contains s- and t-channel seperation

## File: DataAna2_Ex3_Abgabe
* Start z0-env python environment and open jupyter notebook from there
* Import all package with first cell
* Import Data. WARNING: Change string 'path_data' to the directory of daten_1.root on your machine
* You dont need it import the luminosity file. A copy of that is hard coded at the top
* Execute all other cells from top to bottom.

Headings:
* 'Data Import': Imports Data from local path. Pls change relative path 'path_data' to match your directory. Also the luminosity matrix is here.
* 'Particle ID': Contains the previously created algorithms from the MC simulations for particle ID (code and cuts are unchanged)
* 'Inverse Matrix & real number of events': The effiency matrix from MC is inverted and the error matrix of the inverted effiency matrix is calculated via MC Toy experiment.
* 'Getting the event numbers for different energies': We use our identified particles and sort them by the energies that are represented in the luminosity file. We also apply the effeincy matrix.
* 'Calculate cross sections': Calculation of the cross section matrix and the cross section error matrix, as well as a plot of the cross section.
* 'Cross section fit': Do the Breit-Wigner fit to the cross sections and return fit parameter and covariances, as well as plot all fits and only the leptonic fits.
* 'Mass Z Boson': Extract the mass of the Z Boson from fit parameters.
* 'Gamma Z': Extract the width of the Z Boson from fit parameters.
* 'Gamma f': Extract the width of the fermions from fit parameters.
* 'peak cross section for hadronic events': Find peak of cross-section of the hadronic events from the widths an Z0 boson mass
* 'number of light neutrino generations': Calculate number of light neutrino from total and partial decay widths.
* 'peak cross section': Calculate the peak cross section directly from the fit
* 'Confidence level': Calculate the t-values from our values, the values from the instructions and the OPAL data.
* 'percentage deviation': And also the deviations of the values above in percentages.
* 'Exercise 4 - Forward-backward asymmetry': Calculate the AFB from the number of muons scattered in the forward and backward direction by manualy counting them.
* 'Errors': Erros for AFB and the plot with the try to fit.
* 'Weinberg Angle': Get the Weinberg Angle from AFB.
* 'Exercise 5': Calculate the ratios of lepton cross sections at the peaks + confidence level with t-test, as well as the ratios of the hadron cross section and lepton cross section at peak and also the ratios of the branching ratios of hadron to letpon branching ratio.

  
