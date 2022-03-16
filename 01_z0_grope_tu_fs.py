import pandas as pd
import matplotlib.pyplot as plt


path_to_base_dir = 'Data/'

#reading csv input data to pandas df
hadron_df = pd.read_csv(path_to_base_dir+'Z_experiment-hadron.csv')
electron_df = pd.read_csv(path_to_base_dir+'Z_experiment-elektron.csv')
myon_df = pd.read_csv(path_to_base_dir+'Z_experiment-myon.csv')
tauon_df = pd.read_csv(path_to_base_dir+'Z_experiment-tauon.csv')

print(hadron_df)

ecal_sum_e_string = 'ECAL(SumE)'
hcal_sum_e_string = 'HCAL(SumE)'
Pcharged_string = 'Ctrk(SumP)'
Ncharged_string = 'Ctrk(N)'


# plt.style.use(mplhep.style.ATLAS) # You can load ATLAS/CMS/ALICE plot style

# plot histograms for 4 variables for all four particles
plt.hist(hadron_df[ecal_sum_e_string], histtype='step', bins=20, color='C1', label='hadron')
plt.hist(electron_df[ecal_sum_e_string], histtype='step',bins=20, color='C2', label='electron')
plt.hist(myon_df[ecal_sum_e_string], histtype='step',bins=20, color='C3', label='myon')
plt.hist(tauon_df[ecal_sum_e_string], histtype='step',bins=20, color='C4', label='tauon')
plt.title('E_ecal')
plt.ylabel('Counts')
plt.xlabel('E_ecal [GeV]')
plt.legend()
plt.show()

plt.hist(hadron_df[hcal_sum_e_string], histtype='step', bins=10, range=(0,50), color='C1', label='hadron')
plt.hist(electron_df[hcal_sum_e_string], histtype='step',bins=4, color='C2', label='electron')
plt.hist(myon_df[hcal_sum_e_string], histtype='step',bins=20, color='C3', label='myon')
plt.hist(tauon_df[hcal_sum_e_string], histtype='step',bins=20, color='C4', label='tauon')
plt.title('E_hcal')
plt.ylabel('Counts')
plt.xlabel('E_hcal [GeV]')
plt.legend()
plt.show()

plt.hist(hadron_df[Pcharged_string], histtype='step', bins=10,range=(0,110), color='C1', label='hadron')
plt.hist(electron_df[Pcharged_string], histtype='step',bins=4, color='C2', label='electron')
plt.hist(myon_df[Pcharged_string], histtype='step',bins=20, color='C3', label='myon')
plt.hist(tauon_df[Pcharged_string], histtype='step',bins=20, color='C4', label='tauon')
plt.title('Pcharged')
plt.ylabel('Counts')
plt.xlabel('Pcharged [GeV]')
plt.legend()
plt.show()

plt.hist(hadron_df[Ncharged_string], histtype='step', bins=10, range=(0,40), color='C1', label='hadron')
plt.hist(electron_df[Ncharged_string], histtype='step',bins=4, color='C2', label='electron')
plt.hist(myon_df[Ncharged_string], histtype='step',bins=20, color='C3', label='myon')
plt.hist(tauon_df[Ncharged_string], histtype='step',bins=20, color='C4', label='tauon')
plt.title('Ncharged')
plt.ylabel('Counts')
plt.xlabel('Ncharged')
plt.legend()
plt.show()

print('done')