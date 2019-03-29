function [aux] = B_aux_matrix(tau,gamma_vector,min_f)

aux=ones(1,length(tau));
      
for gamma_0 = gamma_vector

    [upper_bound,lower_bound] = B(gamma_0,tau,0,min_f);
    aux_n = max(1,size(aux,1));
    aux_new=[];
    for ind=1:aux_n

        aux_new = [aux_new; aux(ind,:).*upper_bound];
        aux_new = [aux_new; aux(ind,:).*lower_bound];

    end
    aux=aux_new;

end