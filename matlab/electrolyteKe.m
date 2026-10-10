function K = electrolyteKe(ce, T)
% ELECTROLYTEKE Compute electrolyte ionic conductivity for LiPF6
%   K = electrolyteKe(ce, T)
%
% Based on Valøen & Reimers, J. Electrochem. Soc. 152(5) A882, 2005.
%
% Args:
%   ce - Electrolyte concentration in mol/m^3 (physical units, dlarray ok)
%   T  - Temperature in K (physical units, dlarray ok)
%
% Returns:
%   K - Conductivity in S/m (divided by 10 as in original code)

    % Scale concentration to mol/L
    ce_s = ce ./ 1000.0;
    
    % Coefficient matrix k(i,j) for i=0,1,2 and j=0,1,2
    % k = [[-10.5, 0.0740, -6.96e-5];
    %      [0.668, -0.0178, 2.80e-5];
    %      [0.494, -8.86e-4, 0.0]];
    
    % Compute polynomial: x = sum_{i,j} k[i,j] * ce^i * T^j
    % Expanded to avoid loops (dlarray compatible):
    % i=0 terms
    x = -10.5 + 0.0740 .* T + (-6.96e-5) .* T.^2;
    % i=1 terms
    x = x + (0.668 + (-0.0178) .* T + 2.80e-5 .* T.^2) .* ce_s;
    % i=2 terms
    x = x + (0.494 + (-8.86e-4) .* T) .* ce_s.^2;
    
    % Conductivity: K = (x^2) * ce_scaled / 10
    K = (x.^2) .* ce_s ./ 10.0;
end
