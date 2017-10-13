# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 13:21:22 2017

@author: Efe
"""
import numpy as np
import matplotlib.pyplot as plt

def build_support(distance, n_spikes, jitter_ratio, jitter_factor):
    
    return(0);


def feasibility_test_mm(tspikes, fc,pattern,K):
    
    return(0)
    
def main():
    repetitions = 2;
    jitter_factor = 100;
    jitter_ratio=0.75;
    fc_vector = np.array([40]); #[30 40 50];
    n_spikes=25;
    K=4;
    distance_vector = [] ;
    successes = [];
    
    for ind_fc in range(fc_vector.size ):
        fc = fc_vector[ind_fc];
        distance_vector.append(np.array([0.64]));#( 0.6:0.01:1 );
        successes.append(np.zeros((distance_vector[ind_fc].size,1)));
        
        for ind_distance in range(distance_vector[ind_fc].size):
            distance = distance_vector[ind_fc][ind_distance]/fc;
            
            for ind_rep in range(repetitions):
                tspikes = build_support(distance, n_spikes, jitter_ratio, jitter_factor);
                aux_pattern = np.random.randn(n_spikes,K);
                pattern = aux_pattern/np.matlib.repmat(np.sqrt(np.sum(np.conj(aux_pattern)*aux_pattern,2 )),1,K);
                successes[ind_fc][ind_distance] = successes[ind_fc][ind_distance]+feasibility_test_mm(tspikes, fc,pattern,K)    
        
        
if __name__ == "__main__" :
    main()