function D = electrolyteDe(ce, T)
% ELECTROLYTEDE Compute electrolyte diffusion coefficient for LiPF6
%   D = electrolyteDe(ce, T)
%
% Based on Valøen & Reimers, J. Electrochem. Soc. 152(5) A882, 2005.
%
% Args:
%   ce - Electrolyte concentration in mol/m^3 (physical units, dlarray ok)
%   T  - Temperature in K (physical units, dlarray ok)
%
% Returns:
%   D - Diffusion coefficient in m^2/s

    % Scale concentration to mol/L
    ce_s = ce ./ 1000.0;
    
    % Coefficients
    % d = [[-4.43, -54.0]; [-0.22, 0.0]]
    T_g0 = 229.0;
    T_g1 = 5.0;
    
    % Glass transition temperature
    T_g = T_g0 + ce_s .* T_g1;
    
    % Polynomial coefficients
    d1 = -4.43 + (-54.0) ./ (T - T_g);
    d2 = -0.22;  % d[1,1] = 0.0, so d2 = d[1,0] + 0/(T-Tg) = -0.22
    
    % Exponent
    x = d1 + d2 .* ce_s;
    
    % Diffusion coefficient: D = 10^x * 1e-4
    % For dlarray compatibility, use exp(x * log(10)) instead of 10.^x
    DD = exp(x .* log(10));
    D = DD .* 1e-4;
end
