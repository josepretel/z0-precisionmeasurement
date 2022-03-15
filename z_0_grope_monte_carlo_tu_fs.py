import numpy as np
import numpy as np
# import seaborn as sns
# import pandas as pd
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import mplhep
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

from helper_definitions import plot_hist_of_arrays


path_to_base_dir = 'Data/'

#read monte carlo files using uproot
ee_data = uproot.open(path_to_base_dir+'ee.root')
mm_data = uproot.open(path_to_base_dir+'mm.root')
tt_data = uproot.open(path_to_base_dir+'tt.root')
qq_data = uproot.open(path_to_base_dir+'qq.root')

ttree_name = 'myTTree'

## Load branches
branches_ee = ee_data[ttree_name].arrays()
branches_mm = mm_data[ttree_name].arrays()
branches_tt = tt_data[ttree_name].arrays()
branches_qq = qq_data[ttree_name].arrays()

# List to go through different variables
variables = ['Pcharged', 'Ncharged', 'E_ecal', 'E_hcal']

### Print list of 'branches' of the TTree (i.e. list of variable names) - is the same for all files
print(ee_data[ttree_name].keys())

# create dictionaries for each deacy product
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



all_dic = [ee_dic, mm_dic, tt_dic, qq_dic]

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

decay_types = ['electron', 'myon', 'tauon', 'hadron']

plot_hist_of_arrays(list_of_arrays=Pcharged,
                    list_of_labels=decay_types,
                    list_of_bins=[1000,1000,1000,1000],
                    title='Pcharged',
                    xrange=(0,120),
                    yrange=(0,1550),
                    xlabel='Pcharged [GeV]'
                    )

plot_hist_of_arrays(list_of_arrays=Ncharged,
                    list_of_labels=decay_types,
                    list_of_bins=[1000,1000,1000,1000],
                    title='Ncharged',
                    xrange=(0,40),
                    yrange=(0,60000),
                    xlabel='Ncharged'
                    )

plot_hist_of_arrays(list_of_arrays=E_ecal,
                    list_of_labels=decay_types,
                    list_of_bins=[1000,1000,1000,1000],
                    title='E_ecal',
                    xrange=(0,120),
                    yrange=(0,2000),
                    xlabel='E_ecal [GeV]'
                    )

plot_hist_of_arrays(list_of_arrays=E_hcal,
                    list_of_labels=decay_types,
                    list_of_bins=[1000,1000,1000,1000],
                    title='E_hcal',
                    xrange=(0,25),
                    yrange=(0,2000),
                    xlabel='E_hcal [GeV]'
                    )

# plot_hist_of_arrays(list_of_arrays=Pcharged,
#                     list_of_labels=decay_types,
#                     list_of_bins=[1000,1000,1000,1000],
#                     title='Pcharged open windows',
#                     # xrange=(0,120),
#                     # yrange=(0,6000),
#                     )



## Define an numpy array for 'Pcharged'

#pchar = ak.to_numpy(branches_ee[var]) # See Docu (https://awkward-array.org/how-to-convert-numpy.html) for more conversions

#print(f"Array of type '{type(pchar)}' defined for '{var}':\n{pchar}")
#print(pchar.max())






print('done')
