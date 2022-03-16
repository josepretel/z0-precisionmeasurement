import numpy as np
import pandas as pd
import uproot
import awkward as ak
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from helper_definitions import plot_hist_of_arrays, s_and_t_channel, s_channel, t_channel

path_to_base_dir = 'Data/'

# read detector data (with uproot) and supplementary detector data (into a df)
detector_data = uproot.open(path_to_base_dir+'daten_1.root')
detector_supplementary_data_df = pd.read_csv(path_to_base_dir+'daten_1.csv')

ttree_name = 'myTTree'
branches_detector_data = detector_data[ttree_name].arrays()

#extract cos(theta) values as an array from detector data
cos_thet = ak.to_numpy(branches_detector_data['cos_thet'])

#plot cos(theta) values, xlim to one, large number of values with number 999 correspond to wrong detections
#TODO: discuss these in the lab report!
bin_content, bin_edges = plot_hist_of_arrays([cos_thet], [1000],
                                             ['Number of events'], yrange=(0,175), xrange=(-1,1),
                                             xlabel=r'$\cos{\theta}$', verbose=False)
                                            # This corresponds to the differential crosssection

bin_mid = 0.5*(bin_edges[1:] + bin_edges[:-1]) #Calculate midpoint of the bars

#def start and end position (relativ to cos(theta) array position) for fit
start_fit = 20
end_fit = -20

#fit cos(theta) with s and t channel fit
coeffs, covariance = curve_fit(s_and_t_channel, bin_mid[start_fit:end_fit], bin_content[start_fit:end_fit],
                                     sigma = np.sqrt(bin_content[start_fit:end_fit]), absolute_sigma = True)

errors_st_fit = np.sqrt(np.diag(covariance)) # determine the uncertainties of the fit parameters

chi_sqrd_fit = np.sum((bin_content[start_fit:end_fit]-s_and_t_channel(bin_mid[start_fit:end_fit], *coeffs))**2
                      /np.sqrt(bin_content[start_fit:end_fit])**2)/len(bin_content[start_fit:end_fit])  # Calculate chi squared/dof for the fit


'''Here we picture the distribution of the measured values of cos(theta). Around 6/7 of the data points had a default 
value of 999, which were cut off in this histogramm. The fit was not performed over the whole range of [-1, 1]
 in order to neglect disturbing values at the boundaries and get a decent fit. We obtained the '''


print('done')