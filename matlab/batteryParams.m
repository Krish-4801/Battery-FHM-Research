function bp = batteryParams()
% BATTERYPARAMS Returns a struct with all battery parameters for the PINN model.
%   bp = batteryParams() initializes and returns the parameters.

    % Cell dimensions and properties
    bp.A_cell = 0.1037; % m^2, cell cross-sectional area
    bp.L_neg = 53.2e-6; % m
    bp.L_sep = 24.7e-6; % m
    bp.L_pos = 39.9e-6; % m

    % Maximum concentrations
    bp.cs_neg_max = 27088; % mol/m^3
    bp.cs_pos_max = 48700; % mol/m^3
    bp.ce_init = 1200; % mol/m^3

    % Diffusion coefficients
    bp.Ds_neg_eff = 1.4e-11; % m^2/s
    bp.Ds_pos_eff = 1.4e-11; % m^2/s

    % Volume fractions
    bp.eps_n = 0.6260; % active material volume fraction, negative
    bp.eps_p = 0.5741; % active material volume fraction, positive
    bp.eps_en = 0.2979; % electrolyte volume fraction, negative
    bp.eps_ep = 0.3591; % electrolyte volume fraction, positive
    bp.eps_sep = 0.35; % electrolyte volume fraction, separator

    % Reaction rates
    bp.kn_star = 180; % A/mol
    bp.kp_star = 90; % A/mol

    % Other properties
    bp.R_c = 0.0268; % ohm
    bp.xn_init = 0.7916;
    bp.xp_init = 0.3494;
    bp.sigma_neg = 100; % S/m
    bp.sigma_pos = 3.8; % S/m
    
    % Effective conductivities
    bp.Ks_eff_pos = bp.sigma_pos * 0.513;
    bp.Ks_eff_neg = bp.sigma_neg * 0.584;

    % Constants
    bp.t_plus = 0.363;
    bp.R_gas = 8.314; % J/(mol*K)
    bp.F = 96485; % A*s/mol
    bp.I = 2.27; % A

    % Nondimensionalization scales
    bp.x_s = bp.L_neg + bp.L_sep + bp.L_pos;
    bp.t_s = 3600;
    bp.T_s = 298;
    bp.cs_s = bp.cs_pos_max;
    bp.ce_s = bp.ce_init;
    bp.phis_s = 2.5;
    bp.phie_s = 1.0;
    bp.JLi_s = 0.0001;
end
