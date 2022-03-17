### import necessary libraries
import numpy as np
import awkward as ak


def get_efficiency(data,cuts,relevant_vars):
	efficiency_matrix = np.zeros((4,4))
	error_matrix = np.zeros((4,4))
	
	el={'ee': 0,'mm':1,'qq':2,'tt':3}
	
	### Calculate the efficiency matrix elements and errors
	# store the errors in an extra matrix
	
	# How are the efficiency matrix and the corresponding error calculated?
	# For each selection (z0 -> {ee,mm,qq,tt}) there are known cuts (from looking at te MC data)
	# the calculated efficiency of the distinction between two channels from above (labeled with i and j) will be stored in the (i,j) component of the efficiency matrix eps_(i,j)
	# In relevant_vars the variables, which are needed for a proper distinction (selected by looking at MC data) are stored for each pair of possible channels
	# The cuts (stored in cuts) are applied to the relevant variables (by masking the awkward arrays)
	# The (estimator of) efficiency is than calculated by
	#                                                        eps = passed (=: k) / total (=:N)
	# How to calculate the errors ?
	# The (estimated) efficiency is the probability of succes with a certain selection. 
	# For succes failure experiments it is known, that the probability to get k times succes out of N trials is
	# 
	#					P(k;N,eps) = binom(N,k) * eps^k *(1-eps)^(N-k)    
	#
	# k is a possible value of a probability variable X which counts the succes times X = sum_{i=1,...,N} Y (distribution of Y = Bernoulli)
	#
	# The maximum likelihood estimator (MLE) is eps = X / N = k / N
	# The variance is V(eps) = V(X/N) = 1/N^2 sum(V(X)) = 1/N^2 V(sum_{i=1,...,N} Y) = 1/ N V(Y) = 1/N eps(1-eps)
	# Summary: eps = k/N, sigma_eps^2 = 1/N eps(1-eps) -> sigma_eps = sqrt(1/N eps(1-eps))
	
	for channel_i,cuts_i in cuts.items():
		for channel_j,rel_vars in relevant_vars[channel_i].items():
			passed = 0
			total = 0
			for var in rel_vars:
				mask0 = data[channel_j][var]>= cuts_i[var][0]
				mask1 = data[channel_j][var] < cuts_i[var][1]
				total +=len(mask0)
				passed += sum(mask0)+sum(mask1)-len(mask0)
			eps = passed / total
			efficiency_matrix[el[channel_i]][el[channel_j]]=eps
			error_matrix[el[channel_i]][el[channel_j]] = np.sqrt(1/total * eps* (1-eps))
			
				
	return efficiency_matrix,error_matrix
