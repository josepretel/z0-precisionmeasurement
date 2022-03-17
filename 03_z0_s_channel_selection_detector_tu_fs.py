import numpy as np
import pandas as pd
import uproot
import awkward as ak
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from helper_definitions import plot_hist_of_arrays, s_and_t_channel, s_channel, t_channel, import_dictionary, apply_cuts

path_to_base_dir = 'Data/'

# read detector data (with uproot) and supplementary detector data (into a df)
detector_data = uproot.open(path_to_base_dir + 'daten_1.root')
detector_supplementary_data_df = pd.read_csv(path_to_base_dir + 'daten_1.csv')

dic_cuts = import_dictionary(path_to_base_dir + 'cuts_final_dict.npy')



ttree_name = 'myTTree'
branches_detector_data = detector_data[ttree_name].arrays()

dic_data = {}  # dictionary for data arrays of variables
variables = ['Pcharged', 'Ncharged', 'E_ecal', 'E_hcal', 'cos_thet']  # required arrays in the analysis
for var in variables:  # fill data dictionary with desired variable arrays
    dic_data[var] = ak.to_numpy(branches_detector_data[var])


# extract cos(theta) values as an array from detector data
cos_thet_all = ak.to_numpy(branches_detector_data['cos_thet'])

# only select events that pass the electronic cuts
masks_variables = apply_cuts(dic_cuts, dic_data, cos_theta=False)
cos_thet = cos_thet_all[masks_variables['ee']]

# plot cos(theta) values, xlim to one, large number of values with number 999 correspond to wrong detections
# TODO: discuss these in the lab report!
bin_content, bin_edges = plot_hist_of_arrays([cos_thet], [500],
                                             ['Number of events'], yrange=(0, 175), xrange=(-1, 1),
                                             xlabel=r'$\cos{\theta}$', verbose=False)
# This corresponds to the differential crosssection

bin_mid = 0.5 * (bin_edges[1:] + bin_edges[:-1])  # Calculate midpoint of the bars

# def start and end position (relative to cos(theta) array position) for fit
start_fit = 20
end_fit = -20

# fit cos(theta) with s and t channel fit
coeffs_st, covariance_st = curve_fit(s_and_t_channel, bin_mid[start_fit:end_fit], bin_content[start_fit:end_fit],
                                     sigma=np.sqrt(bin_content[start_fit:end_fit]), absolute_sigma=True)

errors_st_fit = np.sqrt(np.diag(covariance_st))  # determine the (uncorrelated) uncertainties of the fit parameters

chi_sqrd_fit = np.sum((bin_content[start_fit:end_fit] - s_and_t_channel(bin_mid[start_fit:end_fit], *coeffs_st)) ** 2
                      / np.sqrt(bin_content[start_fit:end_fit]) ** 2) / len(
    bin_content[start_fit:end_fit])  # Calculate chi squared/dof for the fit

print('S=', coeffs_st[0], r'$\pm$', np.sqrt(covariance_st[0][0]), '\n', 'T=', coeffs_st[1], r'$\pm$',
      np.sqrt(covariance_st[1][1]))  # Print results for fit parameters

# define cut for t channel
cut = 0.70

plt.errorbar(bin_mid[start_fit:end_fit], s_and_t_channel(bin_mid[start_fit:end_fit], *coeffs_st), fmt='-',
             label='s and t channel')
plt.errorbar(bin_mid[start_fit:end_fit], s_channel(bin_mid[start_fit:end_fit], coeffs_st[0]), fmt='--',
             label='s channel')
plt.errorbar(bin_mid[start_fit:end_fit], t_channel(bin_mid[start_fit:end_fit], coeffs_st[1]), fmt=':',
             label='t channel')
plt.axvline(cut, label='cut', color='C7', ls='--')
plt.legend()
plt.show()

'''
Here we picture the distribution of the measured values of cos(theta). Around 6/7 of the data points had a default 
value of 999, which were cut off in this histogram. The fit was not performed over the whole range of [-1, 1]
 in order to neglect disturbing values at the boundaries and get a decent fit. We obtained the contributions
 S=11.34 \pm 0.10 of the s-channel and T=0.257 \pm 0.006 of the t-channel. Based on the fit of both those 
  contributions a cut at cos(theta)<=0.7 is chosen to select only s-channel events. However, this also cuts
  some s-channel events with a higher value. To quantify how many we evaluate the following integrals. The total number
  events that we identify as s-channel contribute to the integral of s- and t-channel in the interval [-1, 0.7].
  The actual number of s-channel events can be determined by the integral of only the s-channel contribution between 
  [-1, 1]. The ratio of this determines how we have to correct the number of s-channel events we found.'''


def integral_s_all(S):
    return 8 / 3 * S


def integral_tands_range(S, T):
    return 6443 / 3000 * S + 17 / 6 * T


chosen_as_s = integral_tands_range(*coeffs_st)
actual_s = integral_s_all(coeffs_st[0])

error_chosen_as_s = integral_s_all(errors_st_fit[0])  # simple error propagation

error_actual_s = np.sqrt(
    (6443 / 3000 * errors_st_fit[0]) ** 2 + (17 / 6 * errors_st_fit[1]) ** 2 + 2 * 6443 / 3000 * 17 / 6 *
    covariance_st[0][1])  # error propagation with covariance of fit parameters

correction_schannel = actual_s / chosen_as_s  # ratio between actual and chosen s-channel events
error_correction_schannel = correction_schannel * np.sqrt(
    (error_actual_s / actual_s) ** 2 + (error_chosen_as_s / chosen_as_s) ** 2)  # gaußian error propagagtion

print('done')
