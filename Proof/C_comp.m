function [C_0,C_1,C_2,C_3] = C_comp(gamma,tau_min,n_aux)
p=length(gamma);
% tail contribution
zeta_func = [Inf pi^2/6 1.202057 pi^4/90 1.036927756...
    pi^6/945 1.008349278 pi^8/945 1.0020084 pi^10/93555]; 
aux_sum_tail = zeta_func(p);
for ind=1:(n_aux-2)
    aux_sum_tail = aux_sum_tail- 1./(ind).^p;
end
C_0 = aux_sum_tail.*( (1.1)./ (4*tau_min) )^p ...
           ./ prod(gamma);  
C_1 = aux_sum_tail.*(2*pi)*( (1.1)./ (4*tau_min) )^p ...
           ./ prod(gamma);  
C_2 = aux_sum_tail.*(2*pi).^2*( (1.1)./ (4*tau_min) )^p ...
           ./ prod(gamma);  
C_3 = aux_sum_tail.*(2*pi).^3*( (1.1)./ (4*tau_min) )^p ...
           ./ prod(gamma);  