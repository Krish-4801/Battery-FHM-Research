function icLoss = computeInitialLoss(params, x_dl, t_dl, C_dl, T_dl, ns, bp)
% COMPUTEINITIALLOSS Compute initial condition loss for t=0
%   Applies initial conditions for all battery regions at t=0
%
% Args:
%   params - PINN network parameters
%   x_dl, t_dl, C_dl, T_dl - Batch inputs (dlarray)
%   ns     - Normalization statistics
%   bp     - Battery parameters

    icLoss = dlarray(single(0));
    
    % Find points where t is approximately 0
    t_phys = zscoreDenormalize(t_dl, ns.t_mean, ns.t_std);
    mask_initial = abs(extractdata(t_phys)) < 1e-3;
    idx = find(mask_initial);
    
    if isempty(idx)
        return;
    end
    
    % Subset input dlarrays
    x_ic = x_dl(idx);
    C_ic = C_dl(idx);
    T_ic = T_dl(idx);
    
    % In Python: t_ic = torch.zeros_like(x_ic)
    t_ic = zeros(size(x_ic), 'like', x_ic);
    
    % Forward pass at t=0
    inputs_ic = cat(2, x_ic, t_ic, C_ic, T_ic);
    outputs_ic = forwardPINN(params, inputs_ic);
    
    cs_ic   = outputs_ic(:, 1);
    ce_ic   = outputs_ic(:, 2);
    phie_ic = outputs_ic(:, 3);
    phis_ic = outputs_ic(:, 4);
    
    % Physical x for region classification
    x_phys_ic = zscoreDenormalize(x_ic, ns.x_mean, ns.x_std);
    x_phys_data = extractdata(x_phys_ic);
    
    % OCP values for initial voltage
    ocv_init = U0_p_ic(bp.xp_init) - U0_n_ic(bp.xn_init);
    
    % Precompute normalized initial condition targets:
    target_cs_anode   = single(((bp.xn_init * bp.cs_neg_max / bp.cs_s) - ns.cs_mean) / ns.cs_std);
    target_cs_cathode = single(((bp.xp_init * bp.cs_pos_max / bp.cs_s) - ns.cs_mean) / ns.cs_std);
    target_ce         = single(((bp.ce_init / bp.ce_s) - ns.ce_mean) / ns.ce_std);
    target_phis_cath  = single(((ocv_init / bp.phis_s) - ns.phis_mean) / ns.phis_std);
    
    % Region masks using physical coordinates
    anode_mask     = cast(x_phys_data >= 0 & x_phys_data <= bp.L_neg, 'like', extractdata(x_ic));
    separator_mask = cast(x_phys_data > bp.L_neg & x_phys_data < (bp.L_neg + bp.L_sep), 'like', extractdata(x_ic));
    cathode_mask   = cast(x_phys_data >= (bp.L_neg + bp.L_sep) & x_phys_data <= (bp.L_neg + bp.L_sep + bp.L_pos), 'like', extractdata(x_ic));
    
    % Anode initial losses (Python: phis, phie compared to zeros_like)
    if any(anode_mask)
        icLoss = icLoss + mean(anode_mask .* abs(phis_ic));
        icLoss = icLoss + mean(anode_mask .* abs(phie_ic));
        icLoss = icLoss + mean(anode_mask .* abs(cs_ic - target_cs_anode));
        icLoss = icLoss + mean(anode_mask .* abs(ce_ic - target_ce));
    end
    
    % Separator initial losses
    if any(separator_mask)
        icLoss = icLoss + mean(separator_mask .* abs(phie_ic));
        icLoss = icLoss + mean(separator_mask .* abs(ce_ic - target_ce));
    end
    
    % Cathode initial losses
    if any(cathode_mask)
        icLoss = icLoss + mean(cathode_mask .* abs(phis_ic - target_phis_cath));
        icLoss = icLoss + mean(cathode_mask .* abs(phie_ic));
        icLoss = icLoss + mean(cathode_mask .* abs(cs_ic - target_cs_cathode));
        icLoss = icLoss + mean(cathode_mask .* abs(ce_ic - target_ce));
    end
end

function u = U0_n_ic(x)
    u = 0.1493 + 0.8493*exp(-61.79*x) + 0.3824*exp(-665.8*x) ...
        - exp(39.42*x - 41.92) - 0.03131*atan(25.59*x - 4.099) ...
        - 0.009434*atan(32.49*x - 15.74);
end

function u = U0_p_ic(x)
    u = -10.72*x.^4 + 23.88*x.^3 - 16.77*x.^2 + 2.595*x + 4.563;
end
