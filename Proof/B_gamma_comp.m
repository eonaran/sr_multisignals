function [B_gamma_0_L,B_gamma_0_U,B_gamma_1_L,B_gamma_1_U,B_gamma_2_L,...
    B_gamma_2_U,B_gamma_3_L,B_gamma_3_U,tau_0_0,tau_0_3] = B_gamma_comp(gamma_vector,tau,min_f)

dimensions = size(tau);
tau=tau(:)';

aux_0 = B_aux_matrix(tau,gamma_vector,min_f);
    
B_gamma_0_U = max(aux_0 ,[],1);
B_gamma_0_L = min( aux_0 ,[],1);    

B_gamma_1_U = 0;
B_gamma_1_L = 0;
B_gamma_2_U = 0;
B_gamma_2_L = 0;
B_gamma_3_U = 0;
B_gamma_3_L = 0;
    

for ind = 1:length(gamma_vector)
        
	gamma_0 = gamma_vector(ind);
    [B_gamma_1_L_aux{ind},B_gamma_1_U_aux{ind},B_gamma_2_L_aux{ind},B_gamma_2_U_aux{ind},...
        B_gamma_3_L_aux{ind},B_gamma_3_U_aux{ind},tau_0_0{ind},tau_0_3{ind}] = B_comp(gamma_0,tau,min_f);
    gamma_aux = gamma_vector([1:(ind-1) (ind+1):end]);
    aux_0 = B_aux_matrix(tau,gamma_aux,min_f);
    n_0 = size(aux_0,1);

    aux_1_U = aux_0.*repmat(B_gamma_1_U_aux{ind},n_0,1);
    aux_1_L = aux_0.*repmat(B_gamma_1_L_aux{ind},n_0,1);
    aux_2_U = aux_0.*repmat(B_gamma_2_U_aux{ind},n_0,1);
    aux_2_L = aux_0.*repmat(B_gamma_2_L_aux{ind},n_0,1);
    aux_3_U = aux_0.*repmat(B_gamma_3_U_aux{ind},n_0,1);
    aux_3_L = aux_0.*repmat(B_gamma_3_L_aux{ind},n_0,1);
    
    aux_mat_1 = [aux_1_U; aux_1_L];
    aux_mat_2 = [aux_2_U; aux_2_L];
    aux_mat_3 = [aux_3_U; aux_3_L];
        
    B_gamma_1_U = B_gamma_1_U + max( aux_mat_1 ,[],1);
    B_gamma_1_L = B_gamma_1_L + min( aux_mat_1 ,[],1);
    B_gamma_2_U = B_gamma_2_U + max( aux_mat_2 ,[],1);
    B_gamma_2_L = B_gamma_2_L + min( aux_mat_2 ,[],1);
    B_gamma_3_U = B_gamma_3_U + max( aux_mat_3 ,[],1);
    B_gamma_3_L = B_gamma_3_L + min( aux_mat_3 ,[],1);    
end
    
for ind_1 = 1:(length(gamma_vector)-1)
    
    for ind_2 = (ind_1+1):length(gamma_vector)
            
       gamma_aux = gamma_vector([1:(ind_1-1) (ind_1+1):(ind_2-1) (ind_2+1):end]);

        aux_0 = B_aux_matrix(tau,gamma_aux,min_f);
        n_0 = size(aux_0,1);

        aux_1_2 = aux_0.*repmat(B_gamma_1_U_aux{ind_1}.*B_gamma_1_U_aux{ind_2},n_0,1);
        aux_2_2 = aux_0.*repmat(B_gamma_1_U_aux{ind_1}.*B_gamma_1_L_aux{ind_2},n_0,1);
        aux_3_2 = aux_0.*repmat(B_gamma_1_L_aux{ind_1}.*B_gamma_1_U_aux{ind_2},n_0,1);
        aux_4_2 = aux_0.*repmat(B_gamma_1_L_aux{ind_1}.*B_gamma_1_L_aux{ind_2},n_0,1);

        aux_mat_2 = [aux_1_2; aux_2_2; aux_3_2; aux_4_2];

        B_gamma_2_U = B_gamma_2_U + 2.*max( aux_mat_2 ,[],1);
        B_gamma_2_L = B_gamma_2_L + 2.*min( aux_mat_2 ,[],1);

        aux_1_3 = aux_0.*repmat(B_gamma_1_U_aux{ind_1}...
            .*B_gamma_2_U_aux{ind_2},n_0,1);
        aux_2_3 = aux_0.*repmat(B_gamma_1_U_aux{ind_1}...
            .*B_gamma_2_L_aux{ind_2},n_0,1);
        aux_3_3 = aux_0.*repmat(B_gamma_1_L_aux{ind_1}...
            .*B_gamma_2_U_aux{ind_2},n_0,1);
        aux_4_3 = aux_0.*repmat(B_gamma_1_L_aux{ind_1}...
            .*B_gamma_2_L_aux{ind_2},n_0,1);

        aux_mat_3 = [aux_1_3; aux_2_3; aux_3_3; aux_4_3];

        B_gamma_3_U = B_gamma_3_U + 3.*max( aux_mat_3 ,[],1);
        B_gamma_3_L = B_gamma_3_L + 3.*min( aux_mat_3 ,[],1);

        aux_1_3 = aux_0.*repmat(B_gamma_1_U_aux{ind_2}...
            .*B_gamma_2_U_aux{ind_1},n_0,1);
        aux_2_3 = aux_0.*repmat(B_gamma_1_U_aux{ind_2}...
            .*B_gamma_2_L_aux{ind_1},n_0,1);
        aux_3_3 = aux_0.*repmat(B_gamma_1_L_aux{ind_2}...
            .*B_gamma_2_U_aux{ind_1},n_0,1);
        aux_4_3 = aux_0.*repmat(B_gamma_1_L_aux{ind_2}...
            .*B_gamma_2_L_aux{ind_1},n_0,1);

        aux_mat_3 = [aux_1_3; aux_2_3; aux_3_3; aux_4_3];

        B_gamma_3_U = B_gamma_3_U + 3.*max( aux_mat_3 ,[],1);
        B_gamma_3_L = B_gamma_3_L + 3.*min( aux_mat_3 ,[],1);

    end

end
        
    
for ind_1 = 1:(length(gamma_vector)-2)

    for ind_2 = (ind_1+1):(length(gamma_vector)-1)

        for ind_3 = (ind_2+1):length(gamma_vector)

            gamma_aux = gamma_vector([1:(ind_1-1) (ind_1+1):(ind_2-1)  (ind_2+1):(ind_3-1) (ind_3+1):end]);

            aux_0 = B_aux_matrix(tau,gamma_aux,min_f);
            n_0 = size(aux_0,1);

            aux_1 = aux_0.*repmat(B_gamma_1_U_aux{ind_1}...
                .*B_gamma_1_U_aux{ind_2}.*B_gamma_1_U_aux{ind_3},n_0,1);
            aux_2 = aux_0.*repmat(B_gamma_1_L_aux{ind_1}...
                .*B_gamma_1_U_aux{ind_2}.*B_gamma_1_U_aux{ind_3},n_0,1);
            aux_3 = aux_0.*repmat(B_gamma_1_U_aux{ind_1}...
                .*B_gamma_1_L_aux{ind_2}.*B_gamma_1_U_aux{ind_3},n_0,1);
            aux_4 = aux_0.*repmat(B_gamma_1_U_aux{ind_1}...
                .*B_gamma_1_U_aux{ind_2}.*B_gamma_1_L_aux{ind_3},n_0,1);
            aux_5 = aux_0.*repmat(B_gamma_1_U_aux{ind_1}...
                .*B_gamma_1_L_aux{ind_2}.*B_gamma_1_L_aux{ind_3},n_0,1);
            aux_6 = aux_0.*repmat(B_gamma_1_L_aux{ind_1}...
                .*B_gamma_1_U_aux{ind_2}.*B_gamma_1_L_aux{ind_3},n_0,1);
            aux_7 = aux_0.*repmat(B_gamma_1_L_aux{ind_1}...
                .*B_gamma_1_L_aux{ind_2}.*B_gamma_1_U_aux{ind_3},n_0,1);
            aux_8 = aux_0.*repmat(B_gamma_1_L_aux{ind_1}...
                .*B_gamma_1_L_aux{ind_2}.*B_gamma_1_L_aux{ind_3},n_0,1);

            aux_mat = [aux_1; aux_2; aux_3; aux_4; aux_5; aux_6; aux_7; aux_8];

            B_gamma_3_U = B_gamma_3_U + 6.*max( aux_mat ,[],1);
            B_gamma_3_L = B_gamma_3_L + 6.*min( aux_mat ,[],1);
        end

    end

end

B_gamma_0_U= reshape(B_gamma_0_U',dimensions);
B_gamma_0_L= reshape(B_gamma_0_L',dimensions);
B_gamma_1_U= reshape(B_gamma_1_U',dimensions);
B_gamma_1_L= reshape(B_gamma_1_L',dimensions);
B_gamma_2_U= reshape(B_gamma_2_U',dimensions);
B_gamma_2_L= reshape(B_gamma_2_L',dimensions);
B_gamma_3_U= reshape(B_gamma_3_U',dimensions);
B_gamma_3_L= reshape(B_gamma_3_L',dimensions);