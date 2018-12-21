function [upper_bound,lower_bound] = B(gamma_0,tau,order,min_f)

epsilon = 1e-4;
tau_aux = epsilon:epsilon:(0.5/gamma_0);
    
[upper_bound_far_aux,lower_bound_far_aux] = B_far_0(gamma_0,tau_aux,min_f);
[upper_bound_near_aux,lower_bound_near_aux] = B_near_0(gamma_0,tau_aux,min_f);

aux_ind_0 = find(upper_bound_near_aux>upper_bound_far_aux);
tau_0_0 = tau_aux(aux_ind_0(1));

if order == 0
    
    [upper_bound_far,lower_bound_far] = B_far_0(gamma_0,tau,min_f);
    [upper_bound_near,lower_bound_near] = B_near_0(gamma_0,tau,min_f);
    
    upper_bound = upper_bound_near;
    lower_bound = lower_bound_near;
    upper_bound(tau>tau_0_0) = upper_bound_far(tau>tau_0_0);
    lower_bound(tau>tau_0_0) = lower_bound_far(tau>tau_0_0);
    upper_bound(tau==0) = 1;
    lower_bound(tau==0) = 1;
end

if order == 1
    
    [upper_bound_0,lower_bound_0] = B(gamma_0,tau,0,min_f);
    tau_gamma = gamma_0.*tau;

    upper_bound = gamma_0 .* (( cos( 2.*pi.*tau_gamma ) - lower_bound_0 )...
        .* ( 1 - ( cos( 2.*pi.*tau_gamma ) <= lower_bound_0 ).*pi^2.*tau_gamma.^2./(2*gamma_0^2*min_f^2) ) ./ tau_gamma)...
        - (sin(2.*pi.*tau_gamma)<0).*pi.*sin(2.*pi.*tau_gamma)./min_f;
    lower_bound = gamma_0 .* (( cos( 2.*pi.*tau_gamma ) - upper_bound_0 )...
        .* ( 1 - ( cos( 2.*pi.*tau_gamma ) >= upper_bound_0 ).*pi^2.*tau_gamma.^2./(2*gamma_0^2*min_f^2) ) ./ tau_gamma)...
        - (sin(2.*pi.*tau_gamma)>=0).*pi.*sin(2.*pi.*tau_gamma)./min_f;
    upper_bound(tau==0) = 0;
    lower_bound(tau==0) = 0;
    
end

if order == 2
    
    [upper_bound_0,lower_bound_0] = B(gamma_0,tau,0,min_f);
    tau_gamma = gamma_0.*tau;
    
    h_2_U = 4.*pi^2.* gamma_0.*(( 1 - ( sin(2.*pi.*tau_gamma) < 0 ).*pi^2.*tau_gamma.^2./(2*gamma_0^2*min_f^2) )...
        .*sin(2.*pi.*tau_gamma)./(2.*pi.*tau_gamma) -lower_bound_0)./min_f;
    h_2_L = 4.*pi^2.* gamma_0.*(( 1 - ( sin(2.*pi.*tau_gamma) >= 0 ).*pi^2.*tau_gamma.^2./(2*gamma_0^2*min_f^2) )...
        .*sin(2.*pi.*tau_gamma)./(2.*pi.*tau_gamma) -upper_bound_0)./min_f;
    
    upper_bound = gamma_0^2 .* (-2.*( cos( 2.*pi.*tau_gamma ) - upper_bound_0 )...
        .*( 1 - ( cos( 2.*pi.*tau_gamma ) > upper_bound_0 ).*pi^2.*tau_gamma.^2./(2*gamma_0^2*min_f^2) ).^2 ...
        ./tau_gamma.^2 ...
        -4.*pi^2.*lower_bound_0 ...
        +h_2_U.*(h_2_U>0)); 
    
    lower_bound =gamma_0^2 .* (-2.*( cos( 2.*pi.*tau_gamma ) - lower_bound_0 )...
        .*( 1 - ( cos( 2.*pi.*tau_gamma ) < lower_bound_0 ).*pi^2.*tau_gamma.^2./(2*gamma_0^2*min_f^2) ).^2 ...
        ./tau_gamma.^2 ...
        -4.*pi^2.*upper_bound_0 ...
        +h_2_L.*(h_2_L<0)); 
    
    upper_bound(tau==0) = - 4 .*pi.^2 .*gamma_0^2 /3;
    lower_bound(tau==0) =- 4 .*pi.^2.*gamma_0^2 *(1+1/(min_f*gamma_0)) /3;
    
end

if order==3
    
    [upper_bound_0_aux,lower_bound_0_aux] = B(gamma_0,tau_aux,0,min_f);
    [upper_bound_1_aux,lower_bound_1_aux] = B(gamma_0,tau_aux,1,min_f);
    [upper_bound_2_aux,lower_bound_2_aux] = B(gamma_0,tau_aux,2,min_f);
    
    tau_gamma_aux = gamma_0.*tau_aux;
    
    h_3_near_U_aux = -4 .* pi.^2 .* gamma_0^2 .* ( sin( 2.*pi.*tau_gamma_aux)./(2*pi.*tau_gamma_aux) ... 
        .*(1-(sin( 2.*pi.*tau_gamma_aux)>0).*pi^2.*tau_gamma_aux.^2./(2.*gamma_0^2.*min_f^2))...
        - upper_bound_0_aux ) ./(gamma_0*min_f)... % - 32 * pi.^4 .* gamma_0^2 .* tau_gamma.^2 ./ 30;
        - gamma_0^2 * pi.^2.*(31.3 * pi.^2 .* tau_gamma_aux.^2 - 16 .*pi^4  .* tau_gamma_aux.^4 .* ( 1 + 2./(gamma_0*min_f) )) ...
        .* (1- pi^2.*tau_gamma_aux.^2./( 2.*gamma_0^2.*min_f^2)).^2 ...
        ./ ( 15 .* ( 2 + 1./(gamma_0 * min_f) ) );

    h_3_near_L_aux = -4 .* pi.^2 .* gamma_0^2 .* ( sin( 2.*pi.*tau_gamma_aux)./(2*pi.*tau_gamma_aux) ...
        .*(1-(sin( 2.*pi.*tau_gamma_aux)<0).*pi^2.*tau_gamma_aux.^2./(2.*gamma_0^2.*min_f^2))...
        - lower_bound_0_aux ) ./(gamma_0*min_f)...
        - 32 * pi.^4 .* gamma_0^2 .* tau_gamma_aux.^2 .* ( 1.17 + 5./(2*gamma_0*min_f) ) ...
        ./ ( 30 .* (1-pi^2.*tau_gamma_aux.^2./(6.*gamma_0^2.*min_f^2)));
        
    h_3_far_U_aux=gamma_0 .* ( 1 - (upper_bound_1_aux<0).*pi^2.*tau_gamma_aux.^2./(2*gamma_0^2*min_f^2) )...
        .*upper_bound_1_aux./tau_gamma_aux - lower_bound_2_aux ;
    
    h_3_far_L_aux=gamma_0 .* ( 1 - (lower_bound_1_aux>0).*pi^2.*tau_gamma_aux.^2./(2*gamma_0^2*min_f^2) )...
        .* lower_bound_1_aux./tau_gamma_aux - upper_bound_2_aux ;

    aux_ind_3 = find(h_3_far_L_aux>h_3_near_L_aux);
    tau_0_3 = tau_aux(aux_ind_3(1));
    
    [upper_bound_0,lower_bound_0] = B(gamma_0,tau,0,min_f);
    [upper_bound_1,lower_bound_1] = B(gamma_0,tau,1,min_f);
    [upper_bound_2,lower_bound_2] = B(gamma_0,tau,2,min_f);
    
    tau_gamma = gamma_0.*tau;

    h_3_near_U_a = -4 .* pi.^2 .* gamma_0^2 .* ( sin( 2.*pi.*tau_gamma)./(2*pi.*tau_gamma) ... 
        .*(1-(sin( 2.*pi.*tau_gamma)>0).*pi^2.*tau_gamma.^2./(2.*gamma_0^2.*min_f^2))...
        - B_0_U ) ./(gamma_0*min_f);
    h_3_near_U_b = - gamma_0^2 * pi.^2.*(31.3 * pi.^2 .* tau_gamma.^2 - 16 .*pi^4  .* tau_gamma.^4 .* ( 1 + 2./(gamma_0*min_f) )) ...
    .* (1- pi^2.*tau_gamma.^2./( 2.*gamma_0^2.*min_f^2)).^2 ...
    ./ ( 15 .* ( 2 + 1./(gamma_0 * min_f) ) );

    h_3_near_L_a = -4 .* pi.^2 .* gamma_0^2 .* ( sin( 2.*pi.*tau_gamma)./(2*pi.*tau_gamma) ...
        .*(1-(sin( 2.*pi.*tau_gamma)<0).*pi^2.*tau_gamma.^2./(2.*gamma_0^2.*min_f^2))...
        - B_0_L ) ./(gamma_0*min_f);
    h_3_near_L_b = - 32 * pi.^4 .* gamma_0^2 .* tau_gamma.^2 .* ( 1.17 + 5./(2*gamma_0*min_f) ) ...
    ./ ( 30 .* (1-pi^2.*tau_gamma.^2./(6.*gamma_0^2.*min_f^2)));

    h_3_far_U=gamma_0 .* ( 1 - (B_1_U<0).*pi^2.*tau_gamma.^2./(2*gamma_0^2*min_f^2) )...
        .*B_1_U./tau_gamma - B_2_L ;

    h_3_far_L=gamma_0 .* ( 1 - (B_1_L>0).*pi^2.*tau_gamma.^2./(2*gamma_0^2*min_f^2) )...
        .* B_1_L./tau_gamma - B_2_U ;

    h_3_U = h_3_near_U_a.*(h_3_near_U_a>0) +h_3_near_U_b;
    h_3_L = h_3_near_L_a.*(h_3_near_L_a<0) +h_3_near_L_b;
    h_3_U(tau>tau_0_3) = h_3_far_U(tau>tau_0_3);
    h_3_L(tau>tau_0_3) = h_3_far_L(tau>tau_0_3);
    
    upper_bound =  -gamma_0^2 .* 4.*pi.^2.*lower_bound_1.*(1+(lower_bound_1>0).*(1/min_f-1/(2*gamma_0^2.*min_f^2)))...
        + 2.*gamma_0.*h_3_U .*(1-(h_3_U<0).*pi^2.*tau_gamma.^2./(2*gamma_0^2*min_f^2))./tau_gamma;
    lower_bound = -gamma_0^2 .* 4.*pi.^2.*upper_bound_1.*(1+(upper_bound_1<0).*(1/min_f-1/(2*gamma_0^2*min_f^2)))...
        + 2.*gamma_0.*h_3_L .*(1-(h_3_L>0).*pi^2.*tau_gamma.^2./(2*gamma_0^2*min_f^2))./tau_gamma;
    
    upper_bound(tau==0) = 0;
    lower_bound(tau==0) = 0;
    
end