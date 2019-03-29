function [bound0,bound1,bound2,bound3] = b_bound_comp(gamma_0,tau,min_f)

bound0 = 1./( 2 * pi * gamma_0 .* tau .* (1-pi^2.*tau.^2./(6 * min_f^2)) );

bound1 = (1 + bound0)./ tau ./ (1-pi^2.*tau.^2./(6 * min_f^2));

bound2 = 4*pi^2*gamma_0^2.*bound0.*(1+1/(gamma_0*min_f)) + 2 .* bound1./tau;

bound3 = 4*pi^2*gamma_0^2.*bound1.*(1+1/(gamma_0*min_f))...
        + 2 .* ( bound2 + bound1 ./tau ) ./tau ;