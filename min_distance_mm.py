# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 13:21:22 2017

@author: Efe
"""
import numpy as np
import matplotlib.pyplot as plt

def main():
    repetitions = 2;
    jitter_factor = 100;
    jitter_ratio=0.75;
    fc_vector = 40;#[30 40 50];
    n_spikes=25;
    K=4;
    distance_vector = [] ;
    successes = [];
    
    for ind_fc in range(len(fc_vector)):
        fc = fc_vector[ind_fc];
        distance_vector.append([0.64]);#( 0.6:0.01:1 );
        successes.append(np.zeros((len(distance_vector[ind_fc]),1)));
        
if __name__ == "__main__" :
    main()