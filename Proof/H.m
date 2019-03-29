function [H_pos,H_neg] = H(gamma,n_1,n_2,tau_min,epsilon,min_f)

epsilon_grid = epsilon;
aux_grid=0:epsilon_grid:tau_min/2;
H_pos = zeros(4,length(aux_grid));
H_neg = zeros(4,length(aux_grid));

for ind_1=1:n_1

    u=((ind_1-1/2)*tau_min):epsilon_grid:((ind_1+4)*tau_min);

    [B_gamma_0_L,B_gamma_0_U,B_gamma_1_L,B_gamma_1_U,B_gamma_2_L,...
        B_gamma_2_U,B_gamma_3_L,B_gamma_3_U] = B_gamma_comp(gamma ,u,min_f);

    bound_aux_0 = max(abs(B_gamma_0_L),abs(B_gamma_0_U))+(2*pi)*epsilon;
    bound_aux_1 = max(abs(B_gamma_1_L),abs(B_gamma_1_U))+(2*pi)^2*epsilon;
    bound_aux_2 = max(abs(B_gamma_2_L),abs(B_gamma_2_U))+(2*pi)^3*epsilon;
    bound_aux_3 = max(abs(B_gamma_3_L),abs(B_gamma_3_U))+(2*pi)^4*epsilon;
    
    [b_gamma_0,b_gamma_1,b_gamma_2,b_gamma_3] = b_bound_gamma_comp(gamma,(ind_1+4)*tau_min,min_f);

    ind_aux=find(abs(u-(ind_1+1/2)*tau_min)<1e-10);
    aux_max_bound_aux_0 = max( max(bound_aux_0(ind_aux:end)) , b_gamma_0);
    aux_max_bound_aux_1 = max(max(bound_aux_1(ind_aux:end)) , b_gamma_1);
    aux_max_bound_aux_2 = max(max(bound_aux_2(ind_aux:end)) , b_gamma_2);
    aux_max_bound_aux_3 = max(max(bound_aux_3(ind_aux:end)) , b_gamma_3);

    max_aux_0=cummax_flip([bound_aux_0(1:ind_aux-1) aux_max_bound_aux_0]);
    max_aux_1=cummax_flip([bound_aux_1(1:ind_aux-1) aux_max_bound_aux_1]);
    max_aux_2=cummax_flip([bound_aux_2(1:ind_aux-1) aux_max_bound_aux_2]);
    max_aux_3=cummax_flip([bound_aux_3(1:ind_aux-1) aux_max_bound_aux_3]);

    ind_H=find(abs(u-ind_1*tau_min)<1e-10);
    H_pos(1,:)=H_pos(1,:)+ max_aux_0(ind_H:-1:1);
    H_neg(1,:) =H_neg(1,:)+ max_aux_0(ind_H:end);
    H_pos(2,:)=H_pos(2,:)+ max_aux_1(ind_H:-1:1);
    H_neg(2,:) =H_neg(2,:)+ max_aux_1(ind_H:end);
    H_pos(3,:)=H_pos(3,:)+ max_aux_2(ind_H:-1:1);
    H_neg(3,:) =H_neg(3,:)+ max_aux_2(ind_H:end);
    H_pos(4,:)=H_pos(4,:)+ max_aux_3(ind_H:-1:1);
    H_neg(4,:) =H_neg(4,:)+ max_aux_3(ind_H:end);
    
end

[b_gamma_0,b_gamma_1,b_gamma_2,b_gamma_3] = b_bound_gamma_comp(gamma,((n_1+1/2):(n_2+1/2)).*tau_min,min_f);

[C_0,C_1,C_2,C_3] = C_comp(gamma,tau_min,n_1+n_2);
        
H_pos(1,:)=H_pos(1,:)+ sum(b_gamma_0)+C_0;
H_pos(2,:)=H_pos(2,:)+ sum(b_gamma_1)+C_1;
H_pos(3,:)=H_pos(3,:)+ sum(b_gamma_2)+C_2;
H_pos(4,:)=H_pos(4,:)+sum(b_gamma_3)+C_3;

[b_gamma_0,b_gamma_1,b_gamma_2,b_gamma_3] = b_bound_gamma_comp(gamma,((n_1+1):(n_2+1)).*tau_min,min_f);

H_neg(1,:)=H_neg(1,:)+ sum(b_gamma_0)+C_0;
H_neg(2,:)=H_neg(2,:)+ sum(b_gamma_1)+C_1;
H_neg(3,:)=H_neg(3,:)+ sum(b_gamma_2)+C_2;
H_neg(4,:)=H_neg(4,:)+sum(b_gamma_3)+C_3;
