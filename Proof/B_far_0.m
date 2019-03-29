function [upper_bound,lower_bound] = B_far_0(gamma_0,tau,min_f)

tau_gamma = gamma_0.*tau;
approx0 = sin( 2.* pi .* tau_gamma)./ (2.* pi .* tau_gamma);
    
aux1_1 = 1;
aux1_2 = (1-(pi^2.*tau_gamma.^2)./(2*gamma_0^2*min_f^2))...
    ./ ( 1+1/(2*gamma_0*min_f) );

aux2_1 = 0;
aux2_2 = 1/(2*gamma_0*min_f+1);

aux = [ approx0 .* aux1_1 + aux2_1 .* cos(2.*pi .*tau_gamma );
    approx0 .* aux1_1  + aux2_2 .* cos(2.*pi .*tau_gamma );
    approx0 .* aux1_2 + aux2_1 .* cos(2.*pi .*tau_gamma );
    approx0 .* aux1_2 + aux2_2 .* cos(2.*pi .*tau_gamma );
    ];

upper_bound = max( aux,[],1);
lower_bound = min( aux,[],1);