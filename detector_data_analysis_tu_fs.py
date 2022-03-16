import numpy as np
import pandas as pd
import uproot
import awkward as ak
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from helper_definitions import plot_hist_of_arrays

path_to_base_dir = 'Data/'

detector_data = uproot.open(path_to_base_dir+'daten_1.root')
detector_supplementary_data_df = pd.read_csv(path_to_base_dir+'daten_1.csv')

ttree_name = 'myTTree'
branches_detector_data = detector_data[ttree_name].arrays()



'''Here we picture the distribution of the measured values of cos(theta). Around 6/7 of the data points had a default 
value of 999, which were cut off in this histogramm. The fit was not performed over the whole range of [-1, 1]
 in order to neglect disturbing values at the boundaries and get a decent fit. We obtained the '''


print('done')