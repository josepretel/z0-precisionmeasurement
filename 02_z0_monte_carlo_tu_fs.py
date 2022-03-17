import uproot
import awkward as ak
import mplhep
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from helper_definitions import plot_hist_of_arrays, get_efficiency_matrix, save_matrix_as_csv, save_dictionary

path_to_base_dir = 'Data/'

# read monte carlo files using uproot
ee_data = uproot.open(path_to_base_dir + 'ee.root')
mm_data = uproot.open(path_to_base_dir + 'mm.root')
tt_data = uproot.open(path_to_base_dir + 'tt.root')
qq_data = uproot.open(path_to_base_dir + 'qq.root')

ttree_name = 'myTTree'

## Load branches into arrays
branches_ee = ee_data[ttree_name].arrays()
branches_mm = mm_data[ttree_name].arrays()
branches_tt = tt_data[ttree_name].arrays()
branches_qq = qq_data[ttree_name].arrays()

# create a list to go through different variables
variables = ['Pcharged', 'Ncharged', 'E_ecal', 'E_hcal']

### Print list of 'branches' of the TTree (i.e. list of variable names) - is the same for all files
print(ee_data[ttree_name].keys())

# create dictionaries for each decay product
ee_dic = {}
mm_dic = {}
tt_dic = {}
qq_dic = {}

# fill dictionaries
for var in variables:
    ee_dic[var] = ak.to_numpy(branches_ee[var])
    mm_dic[var] = ak.to_numpy(branches_mm[var])
    tt_dic[var] = ak.to_numpy(branches_tt[var])
    qq_dic[var] = ak.to_numpy(branches_qq[var])

# combine all dictionarys into one large dictionary
all_dic = [ee_dic, mm_dic, tt_dic, qq_dic]

# Create arrays for each variable which has the data for all 4 decay types
Pcharged = []
Ncharged = []
E_ecal = []
E_hcal = []

for dic in all_dic:
    Pcharged.append(dic[variables[0]])
    Ncharged.append(dic[variables[1]])
    E_ecal.append(dic[variables[2]])
    E_hcal.append(dic[variables[3]])

# print(f"Minimum value of Pcharged: ({pchar.min()}) {np.nanmin(pchar)}")
# print(f"Maximum value of {var}: ({pchar.max()}) {np.nanmax(pchar)}")
# plt.hist(Pcharged[0])
# plt.show()

# create an array of all decay types investigated
decay_types = ['electron', 'myon', 'tauon', 'hadron']

# plot histogams for all four variables with all four decay types per variable
# use an adequate binning, and x/y range
# the plot was done by creating a plot function, see documentation of plot function for further info
plot_hist_of_arrays(list_of_arrays=Pcharged,
                    list_of_labels=decay_types,
                    list_of_bins=[1000, 1000, 1000, 1000],
                    title='Pcharged',
                    xrange=(0, 120),
                    yrange=(0, 1550),
                    xlabel='Pcharged [GeV]'
                    )
plot_hist_of_arrays(list_of_arrays=Ncharged,
                    list_of_labels=decay_types,
                    list_of_bins=[1000, 1000, 1000, 1000],
                    title='Ncharged',
                    xrange=(0, 40),
                    yrange=(0, 60000),
                    xlabel='Ncharged'
                    )
plot_hist_of_arrays(list_of_arrays=E_ecal,
                    list_of_labels=decay_types,
                    list_of_bins=[1000, 1000, 1000, 1000],
                    title='E_ecal',
                    xrange=(0, 120),
                    yrange=(0, 2000),
                    xlabel='E_ecal [GeV]'
                    )
plot_hist_of_arrays(list_of_arrays=E_hcal,
                    list_of_labels=decay_types,
                    list_of_bins=[1000, 1000, 1000, 1000],
                    title='E_hcal',
                    xrange=(0, 25),
                    yrange=(0, 2000),
                    xlabel='E_hcal [GeV]'
                    )

## define cuts

# create dictionary for cuts definition
cuts_initial = {'ee': {}, 'mm': {}, 'tt': {}, 'qq': {}}

# cuts manually defined by looking at the graphs
cuts_initial['ee'] = {'Ncharged': (0, 6),
                      'Pcharged': (0, 120),
                      'E_ecal': (70, 120),
                      'E_hcal': (0, 10)}
cuts_initial['mm'] = {'Ncharged': (0, 6),
                      'Pcharged': (0, 120),
                      'E_ecal': (0, 15),
                      'E_hcal': (0, 10)}
cuts_initial['tt'] = {'Ncharged': (0, 6),
                      'Pcharged': (0, 60),
                      'E_ecal': (0, 60),
                      'E_hcal': (0, 120)}
cuts_initial['qq'] = {'Ncharged': (6, 60),
                      'Pcharged': (0, 120),
                      'E_ecal': (40, 80),
                      'E_hcal': (0, 120)}

# calculate efficinecy matrix for initial guess
eff_matrix_initial, error_eff_initial = get_efficiency_matrix(all_dic, cuts_initial, variables=variables)
print(f'initial cut guess efficiency matrix:')
print(eff_matrix_initial, error_eff_initial)

# optimise the cuts to obtain the final cuts below:

cuts_final = {'ee': {}, 'mm': {}, 'tt': {}, 'qq': {}}

# cuts optimised by looking at the graphs individually
cuts_final['ee'] = {'Ncharged': (0, 6),
                    'Pcharged': (0, 120),
                    'E_ecal': (70, 120),
                    'E_hcal': (0, 10)}  # we tried reducing to (0, 8), but this only reduced the tau assignment minorly,
# but cut quite a few actual electrons
cuts_final['mm'] = {'Ncharged': (0, 6),
                    'Pcharged': (60, 120),
                    # this was the key to eliminate the tauons that passed. This however cuts all muons at 0, which are 4494, which are 5%, but it was worth because it eliminates almost all tauons. Then we increased the other windows to get the muons back up to 90%
                    'E_ecal': (0, 20),
                    'E_hcal': (0, 15)}
cuts_final['tt'] = {'Ncharged': (0, 6),
                    'Pcharged': (1, 70),
                    # with setting lower bound to 1, most muons (at 0) disappeared, opening to 70 increased tauons a lot
                    'E_ecal': (0, 80),  # opening to to 80 only increased tauons but not the others
                    'E_hcal': (0, 120)}
cuts_final['qq'] = {'Ncharged': (6, 60),
                    'Pcharged': (1, 120),
                    'E_ecal': (35, 90),  # this increased the acceptance of hadrons
                    'E_hcal': (1, 120)}


# calculate efficiency matrix for final, optimal cuts

# save cuts final dictionary
save_dictionary(cuts_final, path_to_base_dir + 'cuts_final_dict.npy')

eff_matrix_final, error_eff_final = get_efficiency_matrix(all_dic, cuts_final, variables=variables)
print(f'final cut guess efficiency matrix:')
print(eff_matrix_final)

## Define an numpy array for 'Pcharged'

# pchar = ak.to_numpy(branches_ee[var]) # See Docu (https://awkward-array.org/how-to-convert-numpy.html) for more conversions

# print(f"Array of type '{type(pchar)}' defined for '{var}':\n{pchar}")
# print(pchar.max())


'''Now the code from Jose to determine in uncertainty of the inverse eff matrix'''
### Number of toy experiments to be done
ntoy = 1000

### Create numpy matrix of list to append elements of inverted toy matrices
inverse_toys = np.empty((4, 4))

# Create toy efficiency matrix out of gaussian-distributed random values
for i in range(0, ntoy, 1):
    toy_matrix = np.zeros((4, 4))
    toy_matrix = np.random.normal(eff_matrix_final, error_eff_final, size=(4, 4))

    ### Invert toy matrix
    inverse_toy = np.linalg.inv(toy_matrix)

    # print(inverse_toys.item(0,0),inverse_toy.item(0,0))
    # Append values
    inverse_toys = np.dstack((inverse_toys, inverse_toy))


# Define gaussian function to fit to the toy distributions:
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))


inverse_errors = np.zeros((4, 4))
inverse_means = np.zeros((4, 4))

# plot all 16 plots for the error matrix
fig = plt.figure(figsize=(20, 10), dpi=80)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)
ax00 = plt.subplot(4, 4, 1)
ax01 = plt.subplot(4, 4, 2)
ax02 = plt.subplot(4, 4, 3)
ax03 = plt.subplot(4, 4, 4)

ax10 = plt.subplot(4, 4, 5)
ax11 = plt.subplot(4, 4, 6)
ax12 = plt.subplot(4, 4, 7)
ax13 = plt.subplot(4, 4, 8)

ax20 = plt.subplot(4, 4, 9)
ax21 = plt.subplot(4, 4, 10)
ax22 = plt.subplot(4, 4, 11)
ax23 = plt.subplot(4, 4, 12)

ax30 = plt.subplot(4, 4, 13)
ax31 = plt.subplot(4, 4, 14)
ax32 = plt.subplot(4, 4, 15)
ax33 = plt.subplot(4, 4, 16)

axes = [[ax00, ax01, ax02, ax03],
        [ax10, ax11, ax12, ax13],
        [ax20, ax21, ax22, ax23],
        [ax30, ax31, ax32, ax33]]

# Adapted ranges to fit/plot gaussian distributions successfully
ranges = [[(1.028, 1.035), (0.00028, 0.0005), (-0.013, -0.008), (-0.0005, 0)],
          [(0, 0.0002), (1.1, 1.115), (-0.012, -0.005), (0, 0.0001)],
          [(-0.016, -0.012), (-0.04, -0.03), (1.045, 1.055), (-0.007, -0.004)],
          [(0, 0.0005), (0.0006, 0.001), (-0.028, -0.024), (1.04, 1.06)]]

# Fill histograms for each inverted matrix coefficient:
for j in range(0, 4, 1):
    for k in range(0, 4, 1):
        print(f'j, i:{j},{k}')
        # Diagonal and off-diagonal terms have different histogram ranges
        hbins, hedges, _ = axes[j][k].hist(inverse_toys[j, k, :], bins=30, range=ranges[j][k], histtype='step',
                                           linewidth=2, label=f'toyhist{j}{k}')
        axes[j][k].legend()

        ## Guess initial parameters of the fit by taking random value from hist and std
        _p0 = [ntoy / 10., np.mean(inverse_toys[j, k, :]), np.std(inverse_toys[j, k, :])]

        # Get the fitted curve
        h_mid = 0.5 * (hedges[1:] + hedges[:-1])  # Calculate midpoints for the fit
        coeffs, _ = curve_fit(gauss, h_mid, hbins, p0=_p0, maxfev=100000)
        h_fit = gauss(h_mid, *coeffs)

        axes[j][k].plot(h_mid, h_fit, label=f'Fit{j}{k}')

        inverse_means[j, k] = coeffs[1]
        inverse_errors[j, k] = abs(coeffs[2])
plt.show()

print(f"Erros for the inverse matrix:\n{inverse_errors}")
inv_eff_matrix = np.linalg.inv(eff_matrix_final)

# saving inverted matrix and corresponding error matrix to use in other python file
save_matrix_as_csv(matrix_array=inv_eff_matrix, path_to_save_csv=path_to_base_dir + 'matrix_invers.csv')
save_matrix_as_csv(matrix_array=inverse_errors, path_to_save_csv=path_to_base_dir + 'matrix_invers_errors.csv')

print('done')
