function bcLoss = computeBoundaryLoss(cs, ce, phis, phie, JLi, cs_x, ce_x, phis_x, phie_x, x_dl, C_dl, ns, bp)
% COMPUTEBOUNDARYLOSS Compute boundary condition losses for the PINN
%   Evaluates BCs at 4 locations: x=0, L_neg, L_neg+L_sep, L_neg+L_sep+L_pos
%
%   Uses spatial derivatives and states already computed in physicsGradients
%   to avoid redundant forward passes and nested autodiff tape issues.
%
% Args:
%   cs, ce, phis, phie, JLi           - Network outputs (dlarray [N x 1])
%   cs_x, ce_x, phis_x, phie_x        - Spatial derivatives (dlarray [N x 1])
%   x_dl, C_dl                        - Normalized inputs (dlarray [N x 1])
%   ns                                - Normalization statistics struct
%   bp                                - Battery parameters struct

    bcLoss = dlarray(single(0));
    
    % Physical coordinates
    x_phys = zscoreDenormalize(x_dl, ns.x_mean, ns.x_std);
    JLi_phys = zscoreDenormalize(JLi, ns.JLi_mean, ns.JLi_std);
    
    sharpness = single(1e7);
    tol = single(1e-4);
    
    % Boundary locations
    boundaries = [0, bp.L_neg, bp.L_neg + bp.L_sep, bp.L_neg + bp.L_sep + bp.L_pos];
    
    for bi = 1:4
        loc = boundaries(bi);
        
        % Smooth mask centered at boundary
        mask = sigmoidFn(sharpness .* (x_phys - (loc - tol))) ...
            .* sigmoidFn(sharpness .* ((loc + tol) - x_phys));
        
        mask_active = (extractdata(mask) > 0.01);
        if ~any(mask_active)
            continue;
        end
        
        mask_w = mask;
        
        % Apply boundary-specific conditions (all using L1/MAE loss)
        switch bi
            case 1  % x = 0 (anode current collector)
                % phis = 0
                bcLoss = bcLoss + mean(mask_w .* abs(phis));
                % Ds_neg_eff * (cs_s / x_s) * (cs_std / x_std) * cs_x = 0
                flux_cs = bp.Ds_neg_eff * (bp.cs_s / bp.x_s) * (ns.cs_std / ns.x_std) .* cs_x;
                bcLoss = bcLoss + mean(mask_w .* abs(flux_cs));
                % dce/dx = 0
                flux_ce = (bp.ce_s / bp.x_s) * (ns.ce_std / ns.x_std) .* ce_x;
                bcLoss = bcLoss + mean(mask_w .* abs(flux_ce));
                % dphie/dx = 0
                electrolyte_conduct_scale = (bp.phie_s / bp.x_s) * (ns.phie_std / ns.x_std);
                electrolyte_current = -electrolyte_conduct_scale .* phie_x;
                bcLoss = bcLoss + mean(mask_w .* abs(electrolyte_current));
                
            case 2  % x = L_neg (anode/separator interface)
                % Ds_neg_eff * (cs_s / x_s) * (cs_std / x_std) * cs_x = (-JLi_s * JLi) / (F * L_neg)
                flux_cs = bp.Ds_neg_eff * (bp.cs_s / bp.x_s) * (ns.cs_std / ns.x_std) .* cs_x;
                target_flux = (-bp.JLi_s .* JLi_phys) ./ (bp.F * bp.L_neg);
                bcLoss = bcLoss + mean(mask_w .* abs(flux_cs - target_flux));
                % dphis/dx = 0
                solid_current = -1.0 .* phis_x;
                bcLoss = bcLoss + mean(mask_w .* abs(solid_current));
                
            case 3  % x = L_neg + L_sep (separator/cathode interface)
                % Ds_neg_eff * (cs_s / x_s) * (cs_std / x_std) * cs_x = (-JLi_s * JLi) / (F * L_pos)
                flux_cs = bp.Ds_neg_eff * (bp.cs_s / bp.x_s) * (ns.cs_std / ns.x_std) .* cs_x;
                target_flux = (-bp.JLi_s .* JLi_phys) ./ (bp.F * bp.L_pos);
                bcLoss = bcLoss + mean(mask_w .* abs(flux_cs - target_flux));
                % dphis/dx = 0
                solid_current = -1.0 .* phis_x;
                bcLoss = bcLoss + mean(mask_w .* abs(solid_current));
                
            case 4  % x = L_neg + L_sep + L_pos (cathode current collector)
                % dcs/dx = 0
                flux_cs = bp.Ds_neg_eff * (bp.cs_s / bp.x_s) * (ns.cs_std / ns.x_std) .* cs_x;
                bcLoss = bcLoss + mean(mask_w .* abs(flux_cs));
                % dce/dx = 0
                flux_ce = (bp.ce_s / bp.x_s) * (ns.ce_std / ns.x_std) .* ce_x;
                bcLoss = bcLoss + mean(mask_w .* abs(flux_ce));
                % dphie/dx = 0
                electrolyte_conduct_scale = (bp.phie_s / bp.x_s) * (ns.phie_std / ns.x_std);
                electrolyte_current = -electrolyte_conduct_scale .* phie_x;
                bcLoss = bcLoss + mean(mask_w .* abs(electrolyte_current));
                % -Ks_eff_pos * (phis_s / x_s) * (phis_std / x_std) * phis_x = -I_app / A_cell
                solid_conduct_scale = bp.Ks_eff_pos * (bp.phis_s / bp.x_s) * (ns.phis_std / ns.x_std);
                solid_current = -solid_conduct_scale .* phis_x;
                target_current = -I_app(C_dl, ns, bp) ./ bp.A_cell;
                bcLoss = bcLoss + mean(mask_w .* abs(solid_current - target_current));
        end
    end
end

function y = sigmoidFn(x)
    y = 1 ./ (1 + exp(-x));
end
