'''
date: 2022-03-20
authors: Felix Riesterer & Patrick Sell (group 20)

We chose to include all the code in one single file, since in the end it's
that many lines and the running time is just a couple of seconds. We hope this
doesn't compromise the readability too much.
'''
import uproot
import awkward as ak
import mplhep
import numpy as np
np.set_printoptions(suppress=True) # suppressing scientific notation
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate

# change these to False if you don't want to see the 
# output or you dont't want the figures to pop up
show_out = True  
showfig = True

path_data = 'data_z0experiment/data/'
path_mc = 'mc_z0experiment/'

file_opal = uproot.open(path_data+'daten_2.root')
file_ee = uproot.open(path_mc+'ee.root')
file_mm = uproot.open(path_mc+'mm.root')
file_tt = uproot.open(path_mc+'tt.root')
file_qq = uproot.open(path_mc+'qq.root')

ttree_name = 'myTTree'

branches = np.array([file_ee[ttree_name].arrays(),
                     file_mm[ttree_name].arrays(),
                     file_tt[ttree_name].arrays(),
                     file_qq[ttree_name].arrays(),
                     file_opal[ttree_name].arrays()],
                     dtype=object)

variables = ['Ncharged', 'Pcharged', 'E_ecal', 'E_hcal', 'cos_thet']

# initialising dictionaries for each mode & detector
dict_ee = {}
dict_mm = {}
dict_tt = {}
dict_qq = {}
dict_opal = {}

# assigning the variables to the dictionaries
for var in variables:
    dict_ee[var] = ak.to_numpy(branches[0][var])
    dict_mm[var] = ak.to_numpy(branches[1][var])
    dict_tt[var] = ak.to_numpy(branches[2][var])
    dict_qq[var] = ak.to_numpy(branches[3][var]) 
    dict_opal[var] = ak.to_numpy(branches[4][var])                              

# defining a dictionary containing all variables of all modes
# (seemed usefull at the time, but we ended up using it only once) ¯\_(ツ)_/¯
dict_all = {'ee': dict_ee, 
            'mm': dict_mm, 
            'tt': dict_tt, 
            'qq': dict_qq, 
            'opal': dict_opal}

# storing the mode-shorthand in a list
modes = list(dict_all.keys())
# storing the particle names in a list (for figure titles, etc.)
names = ['electron', 'muon', 'tauon', 'hadron']

def plot_histogram(bins, range, xlim, mode, var, xlabel):
    '''
    This function is a framework for plotting the MC-data histograms
    '''
    plt.style.use(mplhep.style.ATLAS) 
    plt.figure(figsize=(7,5))
    
    bin_content, bin_edges, _ = plt.hist(dict_all[mode][var], bins=bins,range=range, histtype='step',  linewidth=2, edgecolor='b', hatch='/', label=var)
    mid = 0.5*(bin_edges[1:] + bin_edges[:-1]) 
    
    error_sizes = np.sqrt(bin_content)
    
    plt.errorbar(mid, bin_content, yerr=error_sizes, fmt='none')

    plt.title(f"{mode} - {var}")
    plt.xlim(xlim)
    plt.xlabel(xlabel)
    plt.ylabel('Number of events')
    if showfig:
        plt.show()
    plt.close()
    
# a selection of histograms. Naturally, more will be included in the report
plot_histogram(1000, (0,150), (0, 140), 'ee', 'E_ecal', r'$E_{ecal}$')
plot_histogram(1000, (0,200), (60, 110), 'mm', 'Pcharged', r'$P_{char}$')

# initialising a dictionary for the variable cut-offs
cuts = {'ee': {}, 'mm': {}, 'tt': {}, 'qq': {}}

# these have been determined manually using the MC-data (see MC.ipnb)
# non relevant variables are cut at values we deemed physically sensible
cuts['ee'] = {'Ncharged': (0, 6),
              'Pcharged': (0 , 120),
              'E_ecal': (72, 120),
              'E_hcal': (0, 120)}
cuts['mm'] = {'Ncharged': (0 , 10),
              'Pcharged': (74, 120),
              'E_ecal': (0, 20),
              'E_hcal': (0, 10)}
cuts['tt'] = {'Ncharged': (0, 5),
              'Pcharged': (0, 73),
              'E_ecal': (8, 65),
              'E_hcal': (0, 120)}
cuts['qq'] = {'Ncharged': ( 9, 60),
              'Pcharged': (0 , 120),
              'E_ecal': (0, 120),
              'E_hcal': (0, 120)}

# initialising all sorts of matrices used for further computation
eff = np.ones((4,4))  # efficiency matrix
eff_std = np.ones((4,4))  # uncertainty of the efficiency matrix
a = np.ones((4,4))  # number of events pertaining to a specific cut
b = np.ones((4,4))  # number of total events pre-cut
matrix_masks = np.zeros((4,4), dtype=object)  # matrix containing the actual 
                                              # masks in case they are needed again

# the following may need some more explaining, it doesn't need to be that 
# convoluted but it turned out that way because of how we assigned the branches
# previously, and so on. 
# mymask sec is is the mask corresponding to the above defined cuts. 
# The first for loop iterates through the branches, the second one accesses the
# cut-dicts via the 4 modes ('ee', 'mm', ...).
k = 0
for k in range(4):
    i = 0
    for mode in modes[:-1]:
        mymask = (branches[k][variables[0]] >= cuts[mode][variables[0]][0]) & \
                 (branches[k][variables[0]] <= cuts[mode][variables[0]][1]) & \
                 (branches[k][variables[1]] >= cuts[mode][variables[1]][0]) & \
                 (branches[k][variables[1]] <= cuts[mode][variables[1]][1]) & \
                 (branches[k][variables[2]] >= cuts[mode][variables[2]][0]) & \
                 (branches[k][variables[2]] <= cuts[mode][variables[2]][1]) & \
                 (branches[k][variables[3]] >= cuts[mode][variables[3]][0]) & \
                 (branches[k][variables[3]] <= cuts[mode][variables[3]][1])# & \
                 #(branches[k][variables[4]] >= cuts[mode][variables[4]][0]) & \
                 #(branches[k][variables[4]] <= cuts[mode][variables[4]][1])
        matrix_masks[k][i] = mymask
        a[k][i] = sum(mymask)
        b[k][i] = len(mymask)
        i += 1

angdis = lambda x: ((1+x**2))  # this is the angular distribution of the
                               # s-channel events 
                               
# this is a scaling factor that ensures we re-integrate the s-channel events
# that were cut out by applying the cuts to the angle cos_thet
scaling_factor = integrate.quad(angdis, -0.9, 0.5)[0]/integrate.quad(angdis, -1, 1)[0]

for i in range(0,3):  # this adds the cos_thet cut to the electron mode and the
                      # two off-diagonal matrix elements pertaining to the
                      # other leptonic modes
    mask_thet = (branches[i]['cos_thet'] >= -0.9) & (branches[i]['cos_thet'] <= 0.5)
    newmask = matrix_masks[i][0] * mask_thet
    a[0][i] = sum(newmask)
    b[0][i] = (sum(mask_thet)/scaling_factor)

# the hadronic mode contains no t-channel contribution, so the cos_theta cut
# was not applied. We still have to re-scale this matrix element, though.
b[0][3] = b[0][3]/scaling_factor

eff = a/b  # efficiency matrix, i.e. #(Events after the cut)/(Events before the cut)
eff_std = np.sqrt(eff * (a/b**2 + 1/b))  # uncertainty as per gaussian error propagation

matrix = eff  # renaming for the inversion part
error_matrix = eff_std


# weird bug: if this does not get printed, the gauss fits will fail (???)
print(f"s-channel corrected efficiency matrix:\n{matrix}")
print(f"Errors:\n{error_matrix}")

ntoy = 20000 

### Create numpy matrix of list to append elements of inverted toy matrices
inverse_toys = np.empty((4,4))

# Create toy efficiency matrix out of gaussian-distributed random values
for i in range(0,ntoy,1):
    toy_matrix = np.zeros((4,4))
    toy_matrix = np.random.normal(matrix,error_matrix,size=(4,4))
    
    ### Invert toy matrix
    inverse_toy = np.linalg.inv(toy_matrix)
    
    #print(inverse_toys.item(0,0),inverse_toy.item(0,0))
    # Append values
    inverse_toys = np.dstack((inverse_toys,inverse_toy))
    
# Define gaussian function to fit to the toy distributions:
def gauss(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

inverse_errors = np.zeros((4,4))
inverse_means = np.zeros((4,4))


fig = plt.figure(figsize=(20, 10),dpi=80)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)
ax00 = plt.subplot(4,4,1)
ax01 = plt.subplot(4,4,2)
ax02 = plt.subplot(4,4,3)
ax03 = plt.subplot(4,4,4)

ax10 = plt.subplot(4,4,5)
ax11 = plt.subplot(4,4,6)
ax12 = plt.subplot(4,4,7)
ax13 = plt.subplot(4,4,8)

ax20 = plt.subplot(4,4,9)
ax21 = plt.subplot(4,4,10)
ax22 = plt.subplot(4,4,11)
ax23 = plt.subplot(4,4,12)

ax30 = plt.subplot(4,4,13)
ax31 = plt.subplot(4,4,14)
ax32 = plt.subplot(4,4,15)
ax33 = plt.subplot(4,4,16)

axes = [[ax00,ax01,ax02,ax03],
        [ax10,ax11,ax12,ax13],
        [ax20,ax21,ax22,ax23],
        [ax30,ax31,ax32,ax33]]

# We found it tedious to manually pick the ranges manually so we used the 
# means of the toy events as a middle point and chose the range to be 4
# standard deviations
means = np.empty((4,4))
stds = np.empty((4,4))

for j in range(4):
    for i in range(4):
        means[i][j] = np.mean(inverse_toys[i][j])
        stds[i][j] = np.std(inverse_toys[i][j], ddof=1)

range1 = means - 4 * stds
range2 = means + 4 * stds

# Fill histograms for each inverted matrix coefficient:
for j in range(0,4,1):
    for k in range(0,4,1):
        
        # Diagonal and off-diagonal terms have different histogram ranges
        hbins, hedges, _ = axes[j][k].hist(inverse_toys[j,k,:],bins=500,range=(range1[j][k],range2[j][k]),  histtype='step', linewidth=2, label=f'toyhist{j}{k}')
        axes[j][k].legend()
    #ranges[j][k]
        ## Guess initial parameters of the fit by taking random value from hist and std
        _p0 = [ntoy/10.,np.mean(inverse_toys[j,k,:]),np.std(inverse_toys[j,k,:])]

        # Get the fitted curve
        h_mid = 0.5*(hedges[1:] + hedges[:-1]) #Calculate midpoints for the fit
        coeffs, _ = curve_fit(gauss, h_mid, hbins, p0=_p0, maxfev=100000)
        h_fit = gauss(h_mid, *coeffs)
        
        axes[j][k].plot(h_mid, h_fit,label=f'Fit{j}{k}')

        inverse_means[j,k] = coeffs[1]
        inverse_errors[j,k] = abs(coeffs[2])

# in the following, we did not carry out a correction for background events
# since we deemed the effect to be negligible by examining the distribution
# of the detector data

# reading out the luminosity data
luminosities = pd.read_csv('data_z0experiment/lumi_files/daten_2.lum')
lumi = np.array(luminosities['lumi'])
u_lumi = np.array(luminosities['all'])
com_energies = np.array([88.47, 89.46, 90.22, 91.22, 91.97, 92.96, 93.76])

# these are the masks for the 7 energy intervals 
energies = np.zeros([7], dtype=object)
for i in range(7):
    if i <= 5:
        energies[i] = (branches[4]['E_lep'] >= com_energies[i] / 2 - 0.1) & (branches[4]['E_lep'] <= com_energies[i] / 2 + 0.1)
    energies[6] = (branches[4]['E_lep'] >= com_energies[6] / 2 - 0.1)

# initialising the event matrix, which contains the number of events for 
# the 7 energies for all 4 modes
events = np.ones((7,4))
for j in range(7):
    i=0
    for mode in modes[:-1]:
        if mode == 'ee':  # for the electrons we inlude the cos_thet cut
            mymask = (branches[4][variables[0]] >= cuts[mode][variables[0]][0]) & \
                     (branches[4][variables[0]] <= cuts[mode][variables[0]][1]) & \
                     (branches[4][variables[1]] >= cuts[mode][variables[1]][0]) & \
                     (branches[4][variables[1]] <= cuts[mode][variables[1]][1]) & \
                     (branches[4][variables[2]] >= cuts[mode][variables[2]][0]) & \
                     (branches[4][variables[2]] <= cuts[mode][variables[2]][1]) & \
                     (branches[4][variables[3]] >= cuts[mode][variables[3]][0]) & \
                     (branches[4][variables[3]] <= cuts[mode][variables[3]][1]) & \
                     (branches[4][variables[4]] >= -0.9) & (branches[4][variables[4]] <= 0.5)# & \
        else:  # for the other modes we don't 
            mymask = (branches[4][variables[0]] >= cuts[mode][variables[0]][0]) & \
                     (branches[4][variables[0]] <= cuts[mode][variables[0]][1]) & \
                     (branches[4][variables[1]] >= cuts[mode][variables[1]][0]) & \
                     (branches[4][variables[1]] <= cuts[mode][variables[1]][1]) & \
                     (branches[4][variables[2]] >= cuts[mode][variables[2]][0]) & \
                     (branches[4][variables[2]] <= cuts[mode][variables[2]][1]) & \
                     (branches[4][variables[3]] >= cuts[mode][variables[3]][0]) & \
                     (branches[4][variables[3]] <= cuts[mode][variables[3]][1])
        events[j][i] = sum(mymask * energies[j])
        i+=1
                            # counting events with sufficiently high event 
u_events = np.sqrt(events)  # numbers follow the poisson distribution with
                            # variance = N
                            

ratio = np.zeros((7,4))  
u_ratio = np.zeros((7,4))

for i in range(7):  # correcting for the above determined efficiencies
    ratio[i,:] = inverse_means @ events[i]  # @ ~ np.matmul()

cross_sections = ratio.T  # transposing the matrix because we need it that way
u_cross_sections = np.zeros((4,7))

# there's probably a neater way but here we divide each vector by the luminosities
for i in range(4):
    cross_sections[i,:] = cross_sections[i,:] / lumi
    # uncertainty via simple gaussian error propagation
    u_cross_sections[i] = np.sqrt((inverse_means[i][i] * np.sqrt(events.T[i]) / lumi)**2 + \
                              (inverse_errors[i][i] * events.T[i] / lumi)**2 + \
                              (inverse_means[i][i] * events.T[i] * u_lumi / lumi**2)**2)

# cross section correction
xs_correction = {'leptonic': [0.09, 0.20, 0.36, 0.52, 0.22, -0.01, -0.08],
                 'hadronic': [2.0, 4.3, 7.7, 10.8, 4.7, -0.2, -1.6]}
for i in range(4):
    if i == 3:
        cross_sections[3] += xs_correction['hadronic']
    else:
        cross_sections[i] += xs_correction['leptonic']

def bw(x,a,b,c):
    '''
    Breit-Wigner distribution function
    '''
    return c * (x**2 / ((x**2 - a**2)**2 + (x**4 * b**2 / (a**2))))

def plot_xs(xs, u_xs, mode):
    '''
    This takes the cross sections, errors of the cross sections and modes as 
    input and generates a plot and returns the fit parameters, their errors and
    the peaks of the curve.
    '''
    x = np.linspace(min(com_energies)-0.1, max(com_energies)+0.1, 10001)
    

    popt, pcov = curve_fit(bw, com_energies ,xs, p0=[91, 2.5, 1.5], maxfev=10000, bounds=((0, 0, 0), (np.inf, np.inf, np.inf)))#, sigma=u_xs, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
        
    
    plt.style.use(mplhep.style.ATLAS) # You can load ATLAS/CMS/ALICE plot style 
    plt.figure(figsize=(7,5))
    
    plt.errorbar(com_energies, xs, yerr=u_xs, fmt='none', c='orange')
    
    plt.plot(x, bw(x, *popt), '-')
    plt.plot(com_energies, xs, 'o', ms=3)
    
    plt.title(f'Cross section of {names[mode]}ic events')
    plt.xlabel(r"$\sqrt{s}$ in GeV")
    plt.ylabel(r'$\sigma$ in nb')
    plt.show()
    # this is a little convoluted but it made sense at the time
    return [popt[0], popt[1]], [perr[0], perr[1]], max(xs), popt[2], perr[2]

def t_test(a, b, u_a, u_b=0):
    '''
    This t-test determines the "agreement" between values by comparing their 
    distance to the uncertainties.
    '''
    return abs(b - a) / np.sqrt(u_b**2 + u_a**2)

# Z0 mass and decay widths are stored in a matrix, the third parameter is stored separately
matrix_popt = np.zeros((4,2))
matrix_perr = np.zeros((4,2))
p3 = np.zeros(4)
u_p3 = np.zeros(4)
xs_max = np.zeros(4)

matrix_popt[0][:], matrix_perr[0][:], xs_max[0], p3[0], u_p3[0] = plot_xs(cross_sections[0], u_cross_sections[0], 0)
matrix_popt[1][:], matrix_perr[1][:], xs_max[1], p3[1], u_p3[1] = plot_xs(cross_sections[1], u_cross_sections[1], 1)
matrix_popt[2][:], matrix_perr[2][:], xs_max[2], p3[2], u_p3[2] = plot_xs(cross_sections[2], u_cross_sections[2], 2)
matrix_popt[3][:], matrix_perr[3][:], xs_max[3], p3[3], u_p3[3] = plot_xs(cross_sections[3], u_cross_sections[3], 3)

# these are the litarature values taken from the manual
MZ = 91.1876
u_MZ = 0.0021
dw = 2.4952
u_dw = 0.0023

# this matrix stores the t-values for the Z0 mass and decay width from the fit
t_matrix = np.zeros((4,2))

for i in range(4):
    t_matrix[i][0] = t_test(matrix_popt[i][0], MZ, matrix_perr[i][0], u_MZ)
    t_matrix[i][1] = t_test(matrix_popt[i][1], dw, matrix_perr[i][1], u_dw)

# weighted average of the MZ and dw (weighing by their uncertainties)
M_Z_mean = np.average(matrix_popt.T[0], weights=1/matrix_perr.T[0]**2)
u_M_Z_mean = np.sqrt(sum(matrix_perr.T[0]**2)) / len(matrix_perr.T[0])
dw_mean = np.average(matrix_popt.T[1], weights=1/matrix_perr.T[1]**2)
u_dw_mean = np.sqrt(sum(matrix_perr.T[1]**2)) / len(matrix_perr.T[1])

# the angular distribution of the muonic events for the calculation of the
# forwards-backwards asymmetry. No need to filter for energies here.
cos_thet_mc = branches[1]['cos_thet']

# extracting the muonic events from the opal data via the muonic cut-offs
muon_mask = (branches[4][variables[0]] >= cuts['mm'][variables[0]][0]) & \
            (branches[4][variables[0]] <= cuts['mm'][variables[0]][1]) & \
            (branches[4][variables[1]] >= cuts['mm'][variables[1]][0]) & \
            (branches[4][variables[1]] <= cuts['mm'][variables[1]][1]) & \
            (branches[4][variables[2]] >= cuts['mm'][variables[2]][0]) & \
            (branches[4][variables[2]] <= cuts['mm'][variables[2]][1]) & \
            (branches[4][variables[3]] >= cuts['mm'][variables[3]][0]) & \
            (branches[4][variables[3]] <= cuts['mm'][variables[3]][1])

# the detector datas angular distribution of the muonic events
# the energies cut singles out the events in the interval around the peak energy
cos_thet_opal = branches[4]['cos_thet'][muon_mask*energies[3]]

# this cut-offs are specific to the detector and are given in the manual
mask_fw = (cos_thet_mc >= 0) & (cos_thet_mc <= 0.95)
mask_bw = (cos_thet_mc >= -0.95) & (cos_thet_mc <= 0)
N_fw = sum(mask_fw)  # number of events in the "forward" direction for the mc data
N_bw = sum(mask_bw)  # number of events in the "backward" direction for the mc data
u_N_fw = np.sqrt(N_fw)  # again, uncertainty as per poisson statistics
u_N_bw = np.sqrt(N_bw)

# forwards-backwards asymmetry
A_fb_mc = (N_fw - N_bw) / (N_fw + N_bw)  
# uncertainty via gaussian error propagation
u_A_fb_mc = np.sqrt((u_N_fw * 2 * N_bw / (N_fw + N_bw)**2)**2 + (u_N_bw * 2 * N_fw / (N_fw + N_bw)**2)**2)

# same thing for the opal data
mask_fw = (cos_thet_opal >= 0) & (cos_thet_opal <= 0.95)
mask_bw = (cos_thet_opal >= -0.95) & (cos_thet_opal <= 0)
N_fw = sum(mask_fw)
N_bw = sum(mask_bw)
u_N_fw = np.sqrt(N_fw)
u_N_bw = np.sqrt(N_bw)
A_fb_opal = (N_fw - N_bw) / (N_fw + N_bw)
u_A_fb_opal = np.sqrt((u_N_fw * 2 * N_bw / (N_fw + N_bw)**2)**2 + (u_N_bw * 2 * N_fw / (N_fw + N_bw)**2)**2)

# radiation correction of the forwards-backwards asymmetry
radiation_correction = [0.021512, 0.019262, 0.016713, 0.018293, 0.030286, 0.062196, 0.093850]
A_fb_mc += radiation_correction[3]
A_fb_opal += radiation_correction[3]

# approximation of the weinberg angle at the peak energy
weinberg_mc = (1 - np.sqrt(A_fb_mc / 3)) * 1 / 4
weinberg_opal = (1 - np.sqrt(A_fb_opal / 3)) * 1 / 4
# uncertainty via gaussian error propagation
u_weinberg_mc = 1/(8*np.sqrt(3)) * u_A_fb_mc/(np.sqrt(A_fb_mc))
u_weinberg_opal = 1/(8*np.sqrt(3)) * u_A_fb_opal/(np.sqrt(A_fb_opal))

# for the following we needed an uncertainty for the peak of the cross section
# curve, which determined via this small toy-experiment
ntoy = 10000
sigma = np.zeros(ntoy)
xs_max_toy = np.zeros(4)
u_xs_max_toy = np.zeros(4)

for j in range(4):
    for i in range(0,ntoy):
        a = np.random.normal(matrix_popt[j][0], matrix_perr[j][0])
        b = np.random.normal(matrix_popt[j][1], matrix_perr[j][1])
        c = np.random.normal(p3[j], u_p3[j])
        sigma[i] = bw(a, a, b, c)
    xs_max_toy[j] = np.mean(sigma)
    u_xs_max_toy[j] = np.std(sigma)

c = 2.99792458e8 # m/s
hbar = 1.054571817e-34 # Js
hbarc = 1.973e-11 # MeV cm

barn_conv = 2.56819e-6 # conversion from nb to GeV^-2 (natural units)

# converting the cross section peaks to suitable units
xs_max_nu = xs_max * barn_conv
u_xs_max = u_xs_max_toy
u_xs_max_nu = u_xs_max * barn_conv

# from eq. (15) in the manual which relates the peak cross section to the
# partial decay widths. We start with the electronic dw since we need it for the
# other modes
pdw_ee = M_Z_mean * dw_mean * np.sqrt(xs_max_nu[0] / (12 * np.pi))

pdw_mm = (M_Z_mean * dw_mean)**2 * xs_max_nu[1] / (12 * np.pi * pdw_ee)
pdw_tt = (M_Z_mean * dw_mean)**2 * xs_max_nu[2] / (12 * np.pi * pdw_ee)
pdw_qq = (M_Z_mean * dw_mean)**2 * xs_max_nu[3] / (12 * np.pi * pdw_ee)

# the uncertainties stem from gaussian error propagation
u_pdw_ee = np.sqrt((pdw_ee * u_M_Z_mean/M_Z_mean)**2 + \
                    (pdw_ee * u_dw_mean / dw_mean)**2 + \
                    (M_Z_mean / (np.sqrt(12 * np.pi)) * u_xs_max_nu[0]/(2 * np.sqrt(xs_max_nu[0])))**2)
u_pdw_mm = np.sqrt((2 * pdw_mm * u_dw_mean / dw_mean)**2 + \
                   (2 * pdw_mm * u_M_Z_mean / M_Z_mean)**2 + \
                   (0.5 * pdw_mm * u_pdw_ee / pdw_ee)**2 + \
                   (pdw_mm * u_xs_max_nu[1])**2)
u_pdw_tt = np.sqrt((2 * pdw_tt * u_dw_mean / dw_mean)**2 + \
                   (2 * pdw_tt * u_M_Z_mean / M_Z_mean)**2 + \
                   (0.5 * pdw_tt * u_pdw_ee / pdw_ee)**2 + \
                   (pdw_tt * u_xs_max_nu[2])**2)
u_pdw_qq = np.sqrt((2 * pdw_qq * u_dw_mean / dw_mean)**2 + \
                   (2 * pdw_qq * u_M_Z_mean / M_Z_mean)**2 + \
                   (0.5 * pdw_qq * u_pdw_ee / pdw_ee)**2 + \
                   (pdw_qq * u_xs_max_nu[2])**2)

# partial decay width of the neutrino events, along with uncertainty (gauss)
pdw_nu = dw_mean - (pdw_ee + pdw_mm + pdw_tt + pdw_qq)
u_pdw_nu = pdw_nu * np.sqrt((u_dw_mean/dw_mean)**2 + \
                            (u_pdw_ee/pdw_ee)**2 + \
                            (u_pdw_mm/pdw_mm)**2 + \
                            (u_pdw_tt/pdw_tt)**2 + \
                            (u_pdw_qq/pdw_qq)**2)
    
# determining the generation of light neutrinos by dividing by the pdw of a neutrino
gen_nu = pdw_nu / 0.1676
u_gen_nu = u_pdw_nu / 0.1676

# testing the lepton universality:
universality_test = [t_test(pdw_ee, pdw_mm, u_pdw_ee, u_pdw_mm), 
                t_test(pdw_ee, pdw_tt, u_pdw_ee, u_pdw_tt), 
                t_test(pdw_mm, pdw_tt, u_pdw_mm, u_pdw_tt)]

# output of the results
if show_out:
    print(f"Means for the inverse matrix:\n{inverse_means}")
    print(f"Errors for the inverse matrix:\n{inverse_errors}")
    print(f"Fit parameters (MZ, dw):\n{matrix_popt}")
    print(f"Uncertainties of fit parameters:\n{matrix_perr}")
    print(f"Deviation from the literature values of the parameters in units of uncertainties:\n{t_matrix}")
    print(f"Weighted mean of the Z0 mass:\n  {M_Z_mean} ± {u_M_Z_mean}")
    print(f"Deviation from literature value ({MZ} ± {u_MZ}) in units of uncertainties:\n  {t_test(M_Z_mean, MZ, u_M_Z_mean, u_MZ)}")
    print(f"Weighted mean of the Z0 decay width:\n  {dw_mean} ± {u_dw_mean}")
    print(f"Deviation from literature value ({dw} ± {u_dw}) in units of uncertainties:\n  {t_test(dw_mean, dw, u_dw_mean, u_dw)}")
    print(f"Forward-backward asymmetry of MC-data and OPAL-detector:\n  {A_fb_mc} ± {u_A_fb_mc}\n  {A_fb_opal} ± {u_A_fb_opal}")
    print(f"Corresponding Weinberg-angles:\n  {weinberg_mc} ± {u_weinberg_mc}\n  {weinberg_opal} ± {u_weinberg_opal}")
    print(f"Deviation from literature value ({0.23122} ± {0.00004}) in units of uncertainty:\n  {t_test(weinberg_mc, 0.23122, u_weinberg_mc, 0.00004)}\n  {t_test(weinberg_opal, 0.23122, u_weinberg_opal, 0.00004)}")
    print(f"Partial decay widths in GeV:\n  electronic: {pdw_ee}\n  muonic: {pdw_mm}\n  tauonic: {pdw_tt}\n  hadronic:{pdw_qq}")
    print(f"Corresponding uncertainties in GeV:\n  electronic: {u_pdw_ee}\n  muonic: {u_pdw_mm}\n  tauonic: {u_pdw_tt}\n  hadronic:{u_pdw_qq}")
    print(f"Neutrino partial decay width in GeV:\n  {pdw_nu} ± {u_pdw_nu}")
    print(f"Resulting generations of light neutrinos:\n  {gen_nu} ± {u_gen_nu}")
    print(f"Deviations from expected value (3) in units of uncertainties:\n  {t_test(gen_nu, 3, u_gen_nu)}")
    print(f"Devation of leptronic partial decay widths:\n  ee & mm: {universality_test[0]}\n  ee & tt: {universality_test[1]}\n  mm & tt: {universality_test[2]}")
print("\n\n            FIN")