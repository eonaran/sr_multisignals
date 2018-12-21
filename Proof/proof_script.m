% Script to perform computations in the proof of Theorem 2.2 in the paper 
% "Super-resolution of point sources via convex programming" 
% by C. Fernandez-Granda.

clear all
close all

gamma = [1-0.339-0.414 0.414 0.339];

tau_min = 1.26;

min_f = 1e3;
plot_graph = 1;

epsilon_grid = 1e-6
epsilon=epsilon_grid;
epsilon_H = 1e-6;

plot_graphs=1;
n_1=20;
n_2=400;

disp('Computing H, this may take around 25 min')
tic
[H_pos,H_neg] = H(gamma,n_1,n_2,tau_min,epsilon_H,min_f);
toc

tau_near=0:epsilon_grid:(tau_min/2);

disp('Computing B_gamma')
tic
[B_gamma_0_L_near,B_gamma_0_U_near,B_gamma_1_L_near,B_gamma_1_U_near,B_gamma_2_L_near,...
    B_gamma_2_U_near,B_gamma_3_L_near,B_gamma_3_U_near] = B_gamma_comp(gamma ,tau_near,min_f);
toc

K20_upperbound = B_gamma_2_U_near(find(tau_near==0));

IminD0 =  2*H_pos(1,1)

D1 = 2*H_pos(2,1)

K20minD2 =  2*H_pos(3,1)

D2inv = 1 ./ ( abs(K20_upperbound) - K20minD2 )

IminC = IminD0 + D1.^2 .* D2inv

Cinv  = 1 ./ ( 1 - IminC )
D0inv =  1 ./ ( 1 - IminD0 )

alpha_inf = Cinv

K20minS = K20minD2 + D1.^2 .* D0inv

Sinv = 1 ./ ( abs(K20_upperbound) - K20minD2 )
alpha_inf_alternative = D0inv* ( (1+D1^2*Sinv*D0inv))

beta_inf_alternative= Sinv * ( D1 .* D0inv )
beta_inf  = D2inv* D1*Cinv

Im_alpha = IminC*Cinv
Re_alpha_lower = 1 - IminC*Cinv

aux_q2r = Re_alpha_lower .* (B_gamma_2_U_near+(2*pi)^3*epsilon);
aux_q2r(B_gamma_2_U_near>0)=  alpha_inf.* (B_gamma_2_U_near(B_gamma_2_U_near>0)+(2*pi)^3*epsilon);
q2r = aux_q2r + alpha_inf .*(H_pos(3,:)+H_neg(3,:))...
    + beta_inf .*  (max(abs(B_gamma_3_U_near),abs(B_gamma_3_L_near))+(2*pi)^4*epsilon)...
    + beta_inf .* (H_pos(4,:)+H_neg(4,:));

if plot_graph
    
    figure
    plot(tau_near,aux_q2r,'b')
    hold on
    plot(tau_near,alpha_inf .* H_pos(3,:),'--g')
    plot(tau_near,alpha_inf .* H_neg(3,:),'--k')
    plot(tau_near,beta_inf .*  max(abs(B_gamma_3_U_near),abs(B_gamma_3_L_near)),'g')
    plot(tau_near,beta_inf .* H_pos(4,:),'--b')
    plot(tau_near,beta_inf .* H_neg(4,:),'--r')
    plot(tau_near,q2r,'r')
    legend('Bound on main element 2nd der.','Bound on 2nd der. sum 1','Bound on 2nd der. sum 2','Bound on main element 3rd der.','Bound on 3rd der. sum 1','Bound on 3rd der. sum 2','q2r')
end

q2i = Im_alpha .*( max(abs(B_gamma_2_U_near),abs(B_gamma_2_L_near))+(2*pi)^3*epsilon)...
    + alpha_inf .*(H_pos(3,:)+H_neg(3,:))...
    + beta_inf .* ( max(abs(B_gamma_3_U_near),abs(B_gamma_3_L_near))+(2*pi)^4*epsilon)...
    + beta_inf .* (H_pos(4,:)+H_neg(4,:) );

if plot_graph
    
    figure
    plot(tau_near,Im_alpha .*max(abs(B_gamma_2_U_near),abs(B_gamma_2_L_near)),'b')
    hold on
    plot(tau_near,alpha_inf .* H_pos(3,:),'--g')
    plot(tau_near,alpha_inf .* H_neg(3,:),'--k')
    plot(tau_near,beta_inf .*  max(abs(B_gamma_3_U_near),abs(B_gamma_3_L_near)),'g')
    plot(tau_near,beta_inf .* H_pos(4,:),'--b')
    plot(tau_near,beta_inf .* H_neg(4,:),'--r')
    plot(tau_near,q2i,'r')
    legend('Bound on main element 2nd der.','Bound on 2nd der. sum 1','Bound on 2nd der. sum 2','Bound on main element 3rd der.','Bound on 3rd der. sum 1','Bound on 3rd der. sum 2','q2i')
end

qr_lower = Re_alpha_lower .* (B_gamma_0_L_near-(2*pi)*epsilon)...
    - alpha_inf .*(H_pos(1,:)+H_neg(1,:))...
    - beta_inf .* (max(abs(B_gamma_1_U_near),abs(B_gamma_1_L_near))+(2*pi)^2*epsilon)...
    - beta_inf .* (H_pos(2,:)+H_neg(2,:));

if plot_graph
    
    figure
    plot(tau_near,Re_alpha_lower .* B_gamma_0_L_near,'b')
    hold on
    plot(tau_near,alpha_inf .* H_pos(1,:),'--g')
    plot(tau_near,alpha_inf .* H_neg(1,:),'--k')
    plot(tau_near,beta_inf .*  max(abs(B_gamma_1_U_near),abs(B_gamma_1_L_near)),'g')
    plot(tau_near,beta_inf .* H_pos(2,:),'--b')
    plot(tau_near,beta_inf .* H_neg(2,:),'--r')
    plot(tau_near,qr_lower,'r')
    legend('Bound on main element','Bound on sum 1','Bound on sum 2','Bound on main element 1st der.','Bound on 1st der. sum 1','Bound on 1st der. sum 2','qrlower')
end

qi = Im_alpha .* (max(abs(B_gamma_0_U_near),abs(B_gamma_0_L_near))+(2*pi)*epsilon)...
    + alpha_inf .*(H_pos(1,:)+H_neg(1,:) ) ...
    + beta_inf .* (max(abs(B_gamma_1_U_near),abs(B_gamma_1_L_near))+(2*pi)^2*epsilon)...
    + beta_inf .* (H_pos(2,:)+H_neg(2,:) );

if plot_graph
    
    figure
    plot(tau_near,Im_alpha .* max(abs(B_gamma_0_U_near),abs(B_gamma_0_L_near)),'b')
    hold on
    plot(tau_near,alpha_inf .* H_pos(1,:),'--g')
    plot(tau_near,alpha_inf .* H_neg(1,:),'--k')
    plot(tau_near,beta_inf .*  max(abs(B_gamma_1_U_near),abs(B_gamma_1_L_near)),'g')
    plot(tau_near,beta_inf .* H_pos(2,:),'--b')
    plot(tau_near,beta_inf .* H_neg(2,:),'--r')
    plot(tau_near,qi,'r')
    legend('Bound on main element','Bound on sum 1','Bound on sum 2','Bound on main element 1st der.','Bound on 1st der. sum 1','Bound on 1st der. sum 2','qi')
end


qup = alpha_inf .* (max(abs(B_gamma_0_U_near),abs(B_gamma_0_L_near))+(2*pi)*epsilon)...
    + alpha_inf .*(H_pos(1,:)+H_neg(1,:) ) ...
    + beta_inf .* (max(abs(B_gamma_1_U_near),abs(B_gamma_1_L_near))+(2*pi)^2*epsilon)...
    + beta_inf .* (H_pos(2,:)+H_neg(2,:) );

if plot_graph
    
    figure
    plot(tau_near,alpha_inf .* max(abs(B_gamma_0_U_near),abs(B_gamma_0_L_near)),'b')
    hold on
    plot(tau_near,alpha_inf .* H_pos(1,:),'--g')
    plot(tau_near,alpha_inf .* H_neg(1,:),'--k')
    plot(tau_near,beta_inf .*  max(abs(B_gamma_1_U_near),abs(B_gamma_1_L_near)),'g')
    plot(tau_near,beta_inf .* H_pos(2,:),'--b')
    plot(tau_near,beta_inf .* H_neg(2,:),'--r')
    plot(tau_near,qup,'r')
    legend('Bound on main element','Bound on sum 1','Bound on sum 2','Bound on main element 1st der.','Bound on 1st der. sum 1','Bound on 1st der. sum 2','qup')
end

q1abs = alpha_inf .*( max(abs(B_gamma_1_U_near),abs(B_gamma_1_L_near))+(2*pi)^2*epsilon)...
     + alpha_inf .*(H_pos(2,:)+H_neg(2,:) ) ...
    + beta_inf .* (max(abs(B_gamma_2_U_near),abs(B_gamma_2_L_near))+(2*pi)^3*epsilon)...
    + beta_inf .* (H_pos(3,:)+H_neg(3,:) );

if plot_graph
    
    figure
    plot(tau_near,alpha_inf .* max(abs(B_gamma_1_U_near),abs(B_gamma_1_L_near)),'b')
    hold on
    plot(tau_near,alpha_inf .* H_pos(2,:),'--g')
    plot(tau_near,alpha_inf .* H_neg(2,:),'--k')
    plot(tau_near,beta_inf .*  max(abs(B_gamma_2_U_near),abs(B_gamma_2_L_near)),'g')
    plot(tau_near,beta_inf .* H_pos(3,:),'--b')
    plot(tau_near,beta_inf .* H_neg(3,:),'--r')
    plot(tau_near,q1abs,'r')
    legend('Bound on main element 1st der.','Bound on 1st der. sum 1','Bound on 1st der. sum 2','Bound on main element 2nd der.','Bound on 2nd der. sum 1','Bound on 2nd der. sum 2','q1abs')
end

aux_qr = qr_lower .* q2r;
aux_qr(q2r>0)=qup(q2r>0) .* q2r(q2r>0);

second_der_approx = 2*(aux_qr + qi .* q2i + q1abs.^2);
f_approx = 1 + cumsum(cumsum(second_der_approx)).*epsilon_grid.^2;

if plot_graph
    figure
    plot(tau_near,second_der_approx)
    title('approximation of 2nd derivative')

    figure
    plot(tau_near,f_approx)
    title('local bound')
end

ind_middle_aux = find(f_approx>1);
ind_middle = ind_middle_aux(1);
ind_neg_secDer_aux = find(second_der_approx > 0);
ind_neg_secDer = ind_neg_secDer_aux(1);

tau_near(ind_middle)
tau_near(ind_neg_secDer)

if plot_graph

    figure
    plot(tau_near(ind_middle:end),qup(ind_middle:end),'b')
    title('local bound')
end

qup(ind_middle)

tau_mid =(tau_min/2):epsilon_grid:(tau_min);

[B_gamma_0_L_mid,B_gamma_0_U_mid,B_gamma_1_L_mid,B_gamma_1_U_mid,B_gamma_2_L_mid,...
    B_gamma_2_U_mid,B_gamma_3_L_mid,B_gamma_3_U_mid] = B_gamma_comp(gamma ,tau_mid,min_f);
[b_gamma_0,b_gamma_1,b_gamma_2,b_gamma_3] = b_bound_gamma_comp(gamma,tau_min,min_f);

qup_mid = alpha_inf .* max(b_gamma_0,max(max(abs(B_gamma_0_U_mid),abs(B_gamma_0_L_mid))+(2*pi)*epsilon))...
    + alpha_inf .*(H_pos(1,end)+H_neg(1,1) ) ...
    + beta_inf .* max(b_gamma_1,max(max(abs(B_gamma_1_U_mid),abs(B_gamma_1_L_mid))+(2*pi)^2*epsilon))...
    + beta_inf .* (H_pos(2,end)+H_neg(2,1) )
