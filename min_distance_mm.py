# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 13:21:22 2017

@author: Efe
"""
import numpy as np
import matplotlib.pyplot as plt

def dirichlet(f,t):
    if t== 0:
        return(1)
    else:
        return(np.sin((2*f+1)*np.pi*t)/ ((2*f+1)*np.sin(np.pi*t) ) )

def dirichlet_1(f,t):
    if t==0:
        return(0)
    else:
        return(np.pi/np.sin(np.pi*t) * (np.cos((2*f+1)*np.pi*t )-dirichlet(f,t)*np.cos(np.pi*t) ) )
        

def build_support(distance, n_spikes, jitter_ratio, jitter_factor):
    jitter = distance/jitter_factor;    
    tspikes_aux = np.linspace(distance/2, (n_spikes+3/2)*distance, num=n_spikes+2 );
    #offsets = np.cumsum(jitter*np.random.rand((n_spikes,1))* (np.random.rand((n_spikes,1))<jitter_ratio) );
    offsets =0;
    tspikes = tspikes_aux[0:n_spikes] + offsets;
    return(tspikes, (n_spikes+2)*distance); 
    

def kernel_fit(tspikes, distance, fc, pattern, K):
    nspikes = tspikes.size; 
    t = np.linspace(distance/2, (nspikes+3/2)*distance, 10*nspikes );
    t = np.outer(t, np.ones((1, nspikes))) - np.outer(np.ones((t.size, 1)), tspikes);
    dirichlet_vec = np.vectorize(dirichlet, otypes=[np.float])
    M = dirichlet_vec(fc, t);
    dirichlet_1_vec = np.vectorize(dirichlet_1, otypes=[np.float])
    M = np.concatenate((M , dirichlet_1_vec(fc,t)), axis=1 );
    print(np.shape(M))
    return(0)
    
def main():
    repetitions = 2;
    jitter_factor = 100;
    jitter_ratio=0.75;
    fc_vector = np.array([40]); #[30 40 50];
    n_spikes=25;
    K=1;
    distance_vector = [] ;
    successes = [];
    
    for ind_fc in range(fc_vector.size ):
        fc = fc_vector[ind_fc];
        distance_vector.append(np.array([0.64]));#( 0.6:0.01:1 );
        successes.append(np.zeros((distance_vector[ind_fc].size,1)));
        
        for ind_distance in range(distance_vector[ind_fc].size):
            distance = distance_vector[ind_fc][ind_distance]/fc;
            
            for ind_rep in range(repetitions):
                tspikes, interval = build_support(distance, n_spikes, jitter_ratio, jitter_factor);
                aux_pattern = np.random.randn(n_spikes,K)+1j*np.random.randn(n_spikes,K);
                pattern = aux_pattern/np.matlib.repmat(np.sqrt(np.sum(np.conj(aux_pattern)*aux_pattern,1 )),1,K);
                successes[ind_fc][ind_distance] = successes[ind_fc][ind_distance]+kernel_fit(tspikes, distance, fc, pattern,K)    
    
#    print(tspikes)        
if __name__ == "__main__" :
    main()