function [b_gamma_0,b_gamma_1,b_gamma_2,b_gamma_3] = b_bound_gamma_comp(gamma_vector,tau,min_f)

for ind_gamma = 1:length(gamma_vector)
    gamma_0=gamma_vector(ind_gamma);
    [bound0_aux,bound1_aux,bound2_aux,bound3_aux] = b_bound_comp(gamma_0,tau,min_f);
    bound0{ind_gamma}=bound0_aux;
    bound1{ind_gamma}=bound1_aux;
    bound2{ind_gamma}=bound2_aux;
    bound3{ind_gamma}=bound3_aux;
end

b_gamma_0 = 1;
b_gamma_1 = 0;
b_gamma_2 = 0;
b_gamma_3 = 0;

for ind_gamma = 1:length(gamma_vector)
    
    b_gamma_0 = b_gamma_0.*bound0{ind_gamma};
    b_gamma_1 = b_gamma_1+bound1{ind_gamma}.*prod_aux_0(bound0,ind_gamma,length(gamma_vector));
    b_gamma_2 = b_gamma_2+bound2{ind_gamma}.*prod_aux_0(bound0,ind_gamma,length(gamma_vector))...
                + bound1{ind_gamma}.*prod_aux_1(bound0,bound1,ind_gamma,length(gamma_vector));
    b_gamma_3 = b_gamma_3+  bound3{ind_gamma}.*prod_aux_0(bound0,ind_gamma,length(gamma_vector))...
                + 2.*bound2{ind_gamma}.*prod_aux_1(bound0,bound1,ind_gamma,length(gamma_vector))...
                + bound1{ind_gamma}.*prod_aux_2(bound0,bound1,bound2,ind_gamma,length(gamma_vector));

end
end

function res=prod_aux_0(vec,ind,n)
    res=1;
    for j=1:n
        if ~ismember(j,ind) 
            res=res.*vec{j};
        end
    end
end

function res=prod_aux_1(vec0,vec1,ind,n)
    res=0;
    for j=1:n
        if ~ismember(j,ind)
            res=res+vec1{j}.*prod_aux_0(vec0,[ind j],n);
        end
    end
end

function res=prod_aux_2(vec0,vec1,vec2,ind,n)
    res=0;
    for j=1:n
        if ~ismember(j,ind)
            res=res+vec2{j}.*prod_aux_0(vec0,[ind j],n);
            res=res+vec1{j}.*prod_aux_1(vec0,vec1,[ind j],n);
        end
    end

end