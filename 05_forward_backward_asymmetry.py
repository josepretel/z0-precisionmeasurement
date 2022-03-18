import numpy as np
import uproot
import awkward as ak

from helper_definitions import import_dictionary, apply_cuts, plot_hist_of_arrays

path_to_base_dir = 'Data/'

dic_cuts = import_dictionary(path_to_base_dir + 'cuts_final_dict.npy')
dic_data = import_dictionary(path_to_base_dir + 'dic_data.npy')

# read detector data (with uproot) and supplementary detector data (into a df)
mm_data = uproot.open(path_to_base_dir + 'mm.root')
ttree_name = 'myTTree'
## Load branches into arrays
branches_mm = mm_data[ttree_name].arrays()

'''Extract the cos(theta) values for muon events both from the MC and the detector data.'''

# extract cos(theta) values as an array from Monte Carlo data
cos_thet_mc = ak.to_numpy(branches_mm['cos_thet'])

# cos(theta) from detector data filtered for muon events

masks_variables = apply_cuts(dic_cuts, dic_data, cos_theta=False)
cos_thet_all = dic_data['cos_thet']
cos_thet_opal = cos_thet_all[masks_variables['mm']]

# Create dictionary containing the cos(theta) arrays for muons for all 7 com_energies of detector data
com_energies = np.array([88.47, 89.46, 90.22, 91.22, 91.97, 92.96, 93.76])
dic_cos_thet = {}

array_cos_thetas = [0] * 7
for (i, com_energy) in zip(range(7), com_energies):  # identify which events are in the energy invervals and apply masks
    energy_mask = dic_data['com_energy'][masks_variables['mm']] == com_energy
    dic_cos_thet[com_energy] = cos_thet_opal[energy_mask]
    array_cos_thetas[i] = cos_thet_opal[energy_mask]

bin_content_mc, bin_edges_mc = plot_hist_of_arrays([cos_thet_mc], [1000],
                                                   ['Number of events'], yrange=(0, 175), xrange=(-1, 1),
                                                   xlabel=r'$\cos{\theta}$', verbose=True)

# calculate the asymmetry for the MC data (only one energy)
forward_mask_mc = cos_thet_mc < 0.95
forward_mask_mc *= cos_thet_mc > 0
A_forward_mc = sum(forward_mask_mc)

backward_mask_mc = cos_thet_mc > -0.95
backward_mask_mc *= cos_thet_mc < 0
A_backward_mc = sum(backward_mask_mc)

A_fb_mc = (A_forward_mc - A_backward_mc) / (A_forward_mc + A_backward_mc)

# simple Gauss error calculation
A_forward_error_mc = np.sqrt(A_forward_mc)
A_backward_error_mc = np.sqrt(A_backward_mc)
A_fb_mc_error = 2 / ((A_forward_error_mc + A_backward_error_mc) ** 2) * np.sqrt(
    A_forward_error_mc ** 2 * A_backward_error_mc + A_backward_mc ** 2 * A_forward_error_mc)

radiation_corrections = { 'energy' : [ 88.47, 89.46, 90.22, 91.22, 91.97, 92.96, 93.76] ,
                          'correction' : [0.021512, 0.019262, 0.016713, 0.018293, 0.030286, 0.062196, 0.093850]}
A_fb_mc_corrected = A_fb_mc + radiation_corrections['correction'][3]

# Function to determine weinberg-angle out of the forward backward asymmetry
# Simplification only valid for Elep \approx 45.6 GeV
def sin2_w(A):
    return (1 - np.sqrt(A / 3)) * 1 / 4
def error_sin2_w(A, A_error):
    return 1/(8*np.sqrt(3)) * A_error/(np.sqrt(A))

weinberg_mc = sin2_w(A_fb_mc)
weinberg_mc_error = error_sin2_w(A_fb_mc, A_fb_mc_error)


'''For the detector data only consider values |cos(theta)|<0.95 according to https://arxiv.org/abs/hep-ex/0012018 (Zedometry", p.26).  '''
# now analoguously calculate the asymmetry for the OPAL data (7 different energies)
asymmetries_opal = np.zeros(7)
u_asymmetries_opal = np.zeros(7)

for (i, com_energy_var) in zip(range(7), dic_cos_thet):
    bin_content_opal, bin_edges_opal = plot_hist_of_arrays([dic_cos_thet[com_energy_var]], [100],
                                                           ['Number of events'], xrange=(-1, 1),
                                                           xlabel=r'$\cos{\theta}$',
                                                           title=r'OPAL muon events for COM energy {} GeV'.format(
                                                               com_energy_var), verbose=True)
    forward_mask = dic_cos_thet[com_energy_var] < 0.95
    forward_mask *= dic_cos_thet[com_energy_var] > 0
    A_forward = sum(forward_mask)

    backward_mask = dic_cos_thet[com_energy_var] > -0.95
    backward_mask *= dic_cos_thet[com_energy_var] < 0
    A_backward = sum(backward_mask)

    A_fb = (A_forward - A_backward) / (A_forward + A_backward)



    # simple Gauss error calculation
    A_forward_error = np.sqrt(A_forward)
    A_backward_error = np.sqrt(A_backward)
    error_A_fb = 2 / ((A_forward_error + A_backward_error) ** 2) * np.sqrt(
        A_forward_error ** 2 * A_backward_error + A_backward ** 2 * A_forward_error)
    u_asymmetries_opal[i] = error_A_fb
    asymmetries_opal[i] = A_fb + radiation_corrections['correction'][i]

weinberg_opal = sin2_w(asymmetries_opal)
weinberg_opal_error = error_sin2_w(asymmetries_opal, u_asymmetries_opal)


## Tests on lepton universality





print('done')
