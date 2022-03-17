### import necessary libraries
import numpy as np
import awkward as ak


def get_efficiency(data,cuts,relevant_vars):
	efficiency_matrix = np.zeros((4,4))
	error_matrix = np.zeros((4,4))
	
	el={'ee': 0,'mm':1,'qq':2,'tt':3}
	
	### Calculate the efficiency matrix elements 
	for channel_i,cuts_i in cuts.items():
		for channel_j,rel_vars in relevant_vars[channel_i].items():
			passed = 0
			total = 0
			for var in rel_vars:
				mask0 = data[channel_j][var]>= cuts_i[var][0]
				mask1 = data[channel_j][var] < cuts_i[var][1]
				total +=len(mask0)
				passed += sum(mask0)+sum(mask1)-len(mask0)
			efficiency_matrix[el[channel_i]][el[channel_j]]=passed/total
				
	return efficiency_matrix,error_matrix
