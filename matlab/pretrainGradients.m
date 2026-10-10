function [loss, grads] = pretrainGradients(params, x_dl, t_dl, C_dl, T_dl, ...
    cs_data, ce_data, phie_data, phis_data, JLi_data, pw)
% PRETRAINGRADIENTS Gradient function for pretraining (data-only, no physics)
% Inputs:
%   params: network parameters struct
%   x_dl, t_dl, C_dl, T_dl: normalized inputs as dlarray
%   cs_data, ce_data, phie_data, phis_data, JLi_data: normalized target data
%   pw: struct with fields w1-w5 for data loss weights

    % 1. Forward pass
    inputs = cat(2, x_dl, t_dl, C_dl, T_dl);
    outputs = forwardPINN(params, inputs);
    
    % 2. Split outputs
    cs = outputs(:,1);
    ce = outputs(:,2);
    phie = outputs(:,3);
    phis = outputs(:,4);
    JLi = outputs(:,5);
    
    % 3. Compute L1 losses
    loss_cs = mean(abs(cs - cs_data));
    loss_ce = mean(abs(ce - ce_data));
    loss_phie = mean(abs(phie - phie_data));
    loss_phis = mean(abs(phis - phis_data));
    loss_JLi = mean(abs(JLi - JLi_data));
    
    % 4. Total loss
    loss = pw.w1*loss_cs + pw.w2*loss_ce + pw.w3*loss_phie + ...
           pw.w4*loss_phis + pw.w5*loss_JLi;
           
    % 5. Compute gradients
    grads = dlgradient(loss, params);
end
