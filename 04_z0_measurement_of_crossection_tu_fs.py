import numpy as np
import pandas as pd
import uproot
import awkward as ak
from helper_definitions import import_dictionary, import_matrix_from_csv, apply_cuts, breit_wigner_distribution, \
    plot_cme_cross_sec, t_test, get_gamma_ee, get_gamma_ee_error, get_gamma_ff, get_gamma_ff_error, save_dictionary

path_to_base_dir = 'Data/'

# import dic cuts from earlier
dic_cuts = import_dictionary(path_to_base_dir + 'cuts_final_dict.npy')

# Add cos_theta bounds to the cuts dictionary
dic_cuts['ee']['cos_thet'] = (-1, 0.7)  # add s-channel cut to ee events of the dictionary
dic_cuts['mm']['cos_thet'] = (-1, 1000)  # no cut applied to all other events (mm, tt, qq)
dic_cuts['tt']['cos_thet'] = (-1, 1000)
dic_cuts['qq']['cos_thet'] = (-1, 1000)

# import detector and detector supplementary data
detector_data = uproot.open(path_to_base_dir + 'daten_1.root')
detector_supplementary_data_df = pd.read_csv(path_to_base_dir + 'daten_1.csv')

ttree_name = 'myTTree'
branches_detector_data = detector_data[ttree_name].arrays()


dic_data = {}  # dictionary for data arrays of variables
variables = ['Pcharged', 'Ncharged', 'E_ecal', 'E_hcal', 'cos_thet']  # required arrays in the analysis
for var in variables:  # fill data dictionary with desired variable arrays
    dic_data[var] = ak.to_numpy(branches_detector_data[var])

E_lep = ak.to_numpy(branches_detector_data['E_lep'])

# extract the luminosities from the detectos supp. data df into new array
luminosities = np.array(detector_supplementary_data_df['lumi'])

'''Identify which COM energy each event corresponds to'''
com_energies = np.array([88.47, 89.46, 90.22, 91.22, 91.97, 92.96, 93.76])
E_lep_edges = [44, 44.5, 45, 45.5, 45.7, 46, 46.5, 47]  # edges of the energy intervals for each COM energy: manually

energies_datapoints = np.zeros(len(E_lep))
for (i, com_energy) in zip(range(7), com_energies): # identify which events are in the invervals
    mask = True
    mask *= (E_lep >= E_lep_edges[i])
    mask *= (E_lep <= E_lep_edges[i+1])
    energies_datapoints += mask * com_energy

dic_data['com_energy'] = energies_datapoints

#number_of_events = np.array([sum(masks_variables['ee']), sum(masks_variables['mm']), sum(masks_variables['tt']),
 #                            sum(masks_variables[
  #                                   'qq'])])  # Counts the number of events that passed the cuts for each decay mode
u_number_of_events = np.sqrt(number_of_events)
'''The uncertainty of the number of events that pass the filter can be determined by a poisson distribution since the 
numbers are sufficiently high.'''

# importing inverse efficiency matrix and corresponding error matrix from earlier
inv_eff_matrix = import_matrix_from_csv(path_to_base_dir + 'matrix_invers.csv')
inv_eff_matrix_error = import_matrix_from_csv(path_to_base_dir + 'matrix_invers_errors.csv')

branching_ratio = inv_eff_matrix @ number_of_events  # Correct the number of events for each decay mode with the inv efficiency matrix, i.e. the branching ratio

'''Determine the uncertainty of the branching ratio using the error propagation according to Eq. 11 of paper propagation 
of errors. Here the uncertainties of the event fraction vector are uncorrelated since the cuts for each decay mode
can be chosen independently.'''

cov_branch = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        for k in range(4):
            diag_cov_branch[i][j] += (number_of_events[k][j] ** 2 * inv_eff_matrix_error[i][k] ** 2) + \
                                ((inv_eff_matrix[i][k] ** 2) * (u_number_of_events[k][j] ** 2))

errors_branch = np.sqrt(diag_cov_branch)

'''Determine the crosssection by dividing the branching rations by all 7 luminosities'''

cross_sections = branching_ratio / luminosities[None, :]

'''The uncertainty of the cross section can easily be determined by a Gaussian error propagation because there is 
no correlation between the values of the luminosities and the branching ratio.'''

# extract the luminosities "errors all" from the detectors supp. data df into new array
# errors all includes both the statistical and the systematical errors
luminosities_error_all = np.array(detector_supplementary_data_df['all'])

relative_u_branch = errors_branch / branching_ratio
relative_u_lumi = luminosities_error_all/luminosities

relative_u_cross_sections = np.sqrt((relative_u_branch)**2 + (relative_u_lumi[None, :])**2)
u_cross_section = relative_u_cross_sections * cross_sections

print('The crosssections for the 4 decay modes are:', '\n', 'ee:', cross_sections[0], 'with errors', u_cross_section[0])
print('mm:', cross_sections[1], 'with errors', u_cross_section[1])
print('tt:', cross_sections[2], 'with errors', u_cross_section[2])
print('qq:', cross_sections[3], 'with errors', u_cross_section[3])


# XS correction values need to be added to the data
xs_corrections = { 'energy' : [ 88.47, 89.46, 90.22, 91.22, 91.97, 92.96, 93.76] ,
                      'hadronic' : [2.0, 4.3, 7.7, 10.8, 4.7, -0.2, -1.6],
                      'leptonic' : [0.09, 0.20, 0.36, 0.52, 0.22, -0.01, -0.08]}
cross_sections_corrected = np.zeros((4,7))
cross_sections_corrected[0] = cross_sections[0] + xs_corrections['leptonic']
cross_sections_corrected[1] = cross_sections[1] + xs_corrections['leptonic']
cross_sections_corrected[2] = cross_sections[2] + xs_corrections['leptonic']
cross_sections_corrected[3] = cross_sections[3] + xs_corrections['hadronic']



#plotting CME vs cross section:


plot_cme_cross_sec(com_energies, cross_sections[0], cross_section_data_error=u_cross_section[0])
plot_cme_cross_sec(com_energies, cross_sections[1], cross_section_data_error=u_cross_section[1], title=r'$\mu^{+}\mu^{-}$')
plot_cme_cross_sec(com_energies, cross_sections[2], cross_section_data_error=u_cross_section[2], title=r'$\tau^{+}\tau^{-}$')
plot_cme_cross_sec(com_energies, cross_sections[3], cross_section_data_error=u_cross_section[3], title=r'$q\bar{q}$')
'''The relative errorbars of the hadronic crosssections are considerably smaller due to the higher number of hadronic 
events. This reduces the Poisson error.'''


'''
Next a t test is performed to check experiment data with literature values.
Literature is taken from P.A. Zyla et al. (Particle Data Group), Prog. Theor. Exp. Phys. 2020, 083C01 (2020) and 2021 update.
https://pdglive.lbl.gov/DataBlock.action?node=S044W#details-citation-1
'''

MZ_literature = 91.1876
u_MZ = 0.0021
width_literature = 2.4952
u_width = 0.0023
t_test_values = np.zeros((4, 2))

print('done')


