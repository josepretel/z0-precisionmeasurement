import pandas as pd
import uproot



path_to_base_dir = 'Data/'

detector_data = uproot.open(path_to_base_dir+'daten_1.root')

detector_sepplementary_data_df = pd.read_csv(path_to_base_dir+'daten_1.csv')


