import numpy as np
import pandas as pd
import uproot
import awkward as ak
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


path_to_base_dir = 'Data/'

dic_cuts = np.load(path_to_base_dir+'cuts_final_dict.npy', allow_pickle='True').item()
print(f'importing dictionary {path_to_base_dir}cuts_final_dict.npy')

dic_cuts['ee']['cos_thet'] = (-1, 0.7)
dic_cuts['mm']['cos_thet'] = (-1, 0.7)
dic_cuts['tt']['cos_thet'] = (-1, 0.7)
dic_cuts['qq']['cos_thet'] = (-1, 0.7)


print('done')