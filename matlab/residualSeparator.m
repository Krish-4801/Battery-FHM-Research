function [res_ce, res_phie] = residualSeparator(ce, phie, JLi, ...
    ce_t, ce_x, ce_xx, phie_xx, T_norm, ns, bp)
% RESIDUALSEPARATOR Compute FHM PDE residuals for the separator region
%   Only electrolyte equations (no solid phase in separator).
%   All inputs are in normalized space; compatible with dlarray.
%
% Args:
%   ce, phie, JLi              - Network outputs (normalized)
%   ce_t, ce_x, ce_xx, phie_xx - Derivatives in normalized space
%   T_norm                     - Normalized temperature
%   ns                         - Normalization statistics struct
%   bp                         - Battery parameters struct

    % Denormalize to physical space
    T_phys   = zscoreDenormalize(T_norm, ns.T_mean, ns.T_std);
    JLi_phys = zscoreDenormalize(JLi, ns.JLi_mean, ns.JLi_std);
    ce_phys  = zscoreDenormalize(ce, ns.ce_mean, ns.ce_std);

    % Effective electrolyte properties (separator: Bruggeman factor = 0.244)
    Ke_eff = electrolyteKe(ce_phys, T_phys) .* 0.244;
    De_eff = electrolyteDe(ce_phys, T_phys) .* 0.244;

    % Dimensionless coefficients (only those relevant to separator)
    a3 = (De_eff .* bp.t_s) ./ (bp.eps_sep * bp.x_s^2);
    a4 = (bp.JLi_s * bp.t_s) / (bp.F * bp.eps_sep * bp.ce_s);
    a5 = (bp.R_gas * bp.T_s .* T_phys .* bp.t_plus^2 * bp.t_s .* Ke_eff) ...
         ./ (bp.x_s^2 * bp.F^2 * bp.eps_sep * bp.ce_s);
    a6 = (bp.t_plus * bp.t_s .* Ke_eff * bp.phie_s) ...
         ./ (bp.F * bp.x_s^2 * bp.eps_sep * bp.ce_s);
    a8 = (bp.R_gas * bp.T_s .* T_phys * bp.t_plus) ./ (bp.F * bp.phie_s);
    a9 = -(bp.JLi_s * bp.x_s^2) ./ (Ke_eff * bp.phie_s);

    % PDE residuals
    res_ce = (ns.ce_std / ns.t_std) .* ce_t ...
           - a3 .* (ns.ce_std / ns.x_std^2) .* ce_xx ...
           - a4 .* JLi_phys ...
           - a5 .* (((ns.ce_std / ns.x_std^2) .* (ce_xx ./ ce_phys)) ...
                  - ((ns.ce_std / ns.x_std) .* (ce_x ./ ce_phys)).^2) ...
           - a6 .* (ns.phie_std / ns.x_std^2) .* phie_xx;

    res_phie = (ns.phie_std / ns.x_std^2) .* phie_xx ...
             - a8 .* (((ns.ce_std / ns.x_std) .* (ce_x ./ ce_phys)).^2 ...
                    - ((ns.ce_std / ns.x_std^2) .* (ce_xx ./ ce_phys))) ...
             - a9 .* JLi_phys;
end
