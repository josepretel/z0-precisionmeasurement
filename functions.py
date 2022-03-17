### import necessary libraries
import numpy as np
import awkward as ak

def get_efficiency(data,channels,var,cuts):
	efficiency_matrix = np.zeros((4,4))
#	  efficiency matrix, e.g.
#		      ee	 mm 	qq       tt
#		ee     .         .       .       .
#		mm     .         .       .       .
#		qq     .         .       .       .
#		tt     .         .       .       .
		
		
	for i in np.arange(len(channels)):
		for j in np.arange(len(channels)):
			passed = 0
			total = 0
			for var,cut in cuts[channels[i]].items():
				mask0 = data[channels[j]][var] >= cut[0]
				mask1 = data[channels[j]][var] < cut[1]
				total +=len(mask0)
				passed += sum(mask0)+sum(mask1)-len(mask0)
			efficiency_matrix[i][j]=passed/total
				
	
	return efficiency_matrix
