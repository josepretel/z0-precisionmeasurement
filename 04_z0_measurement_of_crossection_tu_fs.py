import numpy as np
import pandas as pd
import uproot
import awkward as ak
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from helper_definitions import import_dictionary, import_matrix_from_csv, apply_cuts

path_to_base_dir = 'Data/'

# import dic cuts from earlier
dic_cuts = import_dictionary(path_to_base_dir + 'cuts_final_dict.npy')

# Add cos_theta bounds to the cuts dictionary
dic_cuts['ee']['cos_thet'] = (-1, 0.7)  # add s-channel cut to the dictionary
dic_cuts['mm']['cos_thet'] = (-1, 1000)
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

luminosities = np.array(detector_supplementary_data_df['lumi'])

'''Identify which COM energy each event corresponds to'''
com_energies = np.array([88.47, 89.46, 90.22, 91.22, 91.97, 92.96, 93.76])
E_lep_edges = [44, 44.5, 45, 45.5, 45.7, 46, 46.5,  47] # edges of the energy intervals for each COM energy

energies_datapoints = np.zeros(len(E_lep))
for (i, com_energy) in zip(range(7), com_energies): # identify which events are in the invervals
    mask = True
    mask *= (E_lep >= E_lep_edges[i])
    mask *= (E_lep <= E_lep_edges[i+1])
    energies_datapoints += mask * com_energy

dic_data['com_energy'] = energies_datapoints






masks = apply_cuts(dic_cuts, dic_data)
mask_cos = dic_data['cos_thet'] <= 0.7
overlap = masks['ee'] * mask_cos

number_of_events = np.array([sum(masks['ee']), sum(masks['mm']), sum(masks['tt']),
                             sum(masks['qq'])])  # Counts the number of events that passed the cuts for each decay mode
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
            cov_branch[i][j] += (inv_eff_matrix[i][k] * inv_eff_matrix[j][k] * (u_number_of_events[k] ** 2))
            if i == j: # Kronecker delta
                cov_branch[i][j] += (number_of_events[k] ** 2 * inv_eff_matrix_error[i][k] ** 2)
            else:
                continue


errors_branch = np.sqrt(np.diag(cov_branch))








'''Determine the crosssection by dividing the branching rations by all 7 luminosities'''





cross_sections = branching_ratio[:, None] / luminosities[None, :]

print('done')
