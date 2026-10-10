function Iapp = I_app(C_norm, ns, bp)
% I_APP Compute applied current in physical units [A]
%   Iapp = I_app(C_norm, ns, bp)
%
% Args:
%   C_norm - Normalized C-rate (dlarray or numeric)
%   ns     - Normalization statistics struct containing C_mean and C_std
%   bp     - Battery parameters struct containing nominal 1C current I (2.27 A)
%
% Returns:
%   Iapp   - Applied current I * C in physical units [A]

    C_phys = zscoreDenormalize(C_norm, ns.C_mean, ns.C_std);
    Iapp = bp.I .* C_phys;
end
