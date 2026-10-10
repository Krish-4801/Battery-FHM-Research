function [loss, grads] = physicsGradients(params, x_dl, t_dl, C_dl, T_dl, ...
    cs_data, ce_data, phie_data, phis_data, JLi_data, ns, bp, w)
% PHYSICSGRADIENTS Evaluates data loss + PDE residuals + BCs + ICs and computes gradients
%
% Inputs:
%   params: network parameter struct
%   x_dl, t_dl, C_dl, T_dl: batch inputs as dlarray [batch x 1]
%   cs_data, ce_data, phie_data, phis_data, JLi_data: target values [batch x 1]
%   ns: normalization statistics struct
%   bp: battery parameters struct
%   w: struct of loss weights (w1 to w8)

    % 1. Forward pass
    inputs = cat(2, x_dl, t_dl, C_dl, T_dl);
    outputs = forwardPINN(params, inputs);

    cs   = outputs(:, 1);
    ce   = outputs(:, 2);
    phie = outputs(:, 3);
    phis = outputs(:, 4);
    JLi  = outputs(:, 5);

    % 2. Data losses (MAE / L1)
    loss_cs   = mean(abs(cs - cs_data));
    loss_ce   = mean(abs(ce - ce_data));
    loss_phie = mean(abs(phie - phie_data));
    loss_phis = mean(abs(phis - phis_data));
    loss_JLi  = mean(abs(JLi - JLi_data));

    % 3. First derivatives with respect to t and x using dlgradient
    cs_t   = dlgradient(sum(cs),   t_dl, 'EnableHigherDerivatives', true);
    cs_x   = dlgradient(sum(cs),   x_dl, 'EnableHigherDerivatives', true);
    ce_t   = dlgradient(sum(ce),   t_dl, 'EnableHigherDerivatives', true);
    ce_x   = dlgradient(sum(ce),   x_dl, 'EnableHigherDerivatives', true);
    phie_x = dlgradient(sum(phie), x_dl, 'EnableHigherDerivatives', true);
    phis_x = dlgradient(sum(phis), x_dl, 'EnableHigherDerivatives', true);

    % 4. Second derivatives with respect to x
    cs_xx   = dlgradient(sum(cs_x),   x_dl, 'EnableHigherDerivatives', true);
    ce_xx   = dlgradient(sum(ce_x),   x_dl, 'EnableHigherDerivatives', true);
    phie_xx = dlgradient(sum(phie_x), x_dl, 'EnableHigherDerivatives', true);
    phis_xx = dlgradient(sum(phis_x), x_dl, 'EnableHigherDerivatives', true);

    % 5. Spatial region identification via smooth masking in physical space
    x_eval = zscoreDenormalize(x_dl, ns.x_mean, ns.x_std);
    sharpness = 1e7;

    % Anode: 0 <= x <= L_neg
    mask_anode = sigmoidMask(x_eval, 0, bp.L_neg, sharpness);
    % Separator: L_neg <= x <= L_neg + L_sep
    mask_sep   = sigmoidMask(x_eval, bp.L_neg, bp.L_neg + bp.L_sep, sharpness);
    % Cathode: L_neg + L_sep <= x <= L_neg + L_sep + L_pos
    mask_pos   = sigmoidMask(x_eval, bp.L_neg + bp.L_sep, bp.L_neg + bp.L_sep + bp.L_pos, sharpness);

    % 6. Compute PDE residuals in each domain
    [r_cs_a, r_ce_a, r_phis_a, r_phie_a] = residualAnode(cs, ce, phis, phie, JLi, ...
        cs_t, cs_xx, ce_t, ce_x, ce_xx, phie_xx, phis_xx, T_dl, ns, bp);

    [r_ce_s, r_phie_s] = residualSeparator(ce, phie, JLi, ...
        ce_t, ce_x, ce_xx, phie_xx, T_dl, ns, bp);

    [r_cs_c, r_ce_c, r_phis_c, r_phie_c] = residualCathode(cs, ce, phis, phie, JLi, ...
        cs_t, cs_xx, ce_t, ce_x, ce_xx, phie_xx, phis_xx, T_dl, ns, bp);

    % 7. Aggregate residual loss
    residualLoss = ...
        mean(mask_anode .* abs(r_cs_a)) + ...
        mean(mask_anode .* abs(r_ce_a)) + ...
        mean(mask_anode .* abs(r_phis_a)) + ...
        mean(mask_anode .* abs(r_phie_a)) + ...
        mean(mask_sep .* abs(r_ce_s)) + ...
        mean(mask_sep .* abs(r_phie_s)) + ...
        mean(mask_pos .* abs(r_cs_c)) + ...
        mean(mask_pos .* abs(r_ce_c)) + ...
        mean(mask_pos .* abs(r_phis_c)) + ...
        mean(mask_pos .* abs(r_phie_c));

    % 8. Boundary and Initial Condition losses
    bcLoss = computeBoundaryLoss(cs, ce, phis, phie, JLi, cs_x, ce_x, phis_x, phie_x, x_dl, C_dl, ns, bp);
    icLoss = computeInitialLoss(params, x_dl, t_dl, C_dl, T_dl, ns, bp);

    % 9. Total weighted physics-informed loss
    loss = w.w1 * residualLoss + ...
           w.w2 * bcLoss + ...
           w.w3 * icLoss + ...
           w.w4 * loss_cs + ...
           w.w5 * loss_ce + ...
           w.w6 * loss_phie + ...
           w.w7 * loss_phis + ...
           w.w8 * loss_JLi;

    % 10. Automatic differentiation for parameter updates
    grads = dlgradient(loss, params);
end

function m = sigmoidMask(x, lower, upper, sharpness)
    sig1 = 1 ./ (1 + exp(-sharpness * (x - lower)));
    sig2 = 1 ./ (1 + exp(-sharpness * (upper - x)));
    m = sig1 .* sig2;
end
