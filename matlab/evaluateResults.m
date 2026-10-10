function evaluateResults(params, ns, bp, dataPath, temperature, c_rate)
% EVALUATERESULTS Compute evaluation metrics and generate discharge/SOC curves
%
% Args:
%   params      - Trained PINN network parameters struct
%   ns          - Normalization statistics struct
%   bp          - Battery parameters struct
%   dataPath    - Path to Data.mat file
%   temperature - Temperature in K (e.g., 298.0)
%   c_rate      - C-rate (e.g., 1)
%
% Generates:
%   - Discharge curve comparison (predicted vs true)
%   - MAE of discharge voltage over time
%   - SOC curve comparison (predicted vs true)
%   - MAE of SOC over time

    fprintf('Evaluating at T=%.0fK (%.0f°C), C=%.1f\n', ...
        temperature, temperature - 273, c_rate);
    
    %% Load raw data for this temperature and C-rate
    data = load(dataPath);
    
    T_str = sprintf('T%.0f', temperature);   % e.g., 'T298'
    C_str = strrep(sprintf('C%g', c_rate), '.', '_'); % e.g., 'C1' or 'C0_5'
    prefix = [T_str '_' C_str];
    
    % Get spatial coordinates
    X_field = [prefix '_X'];
    if ~isfield(data, X_field)
        error('Data for T=%.0f, C=%g not found in Data.mat', temperature, c_rate);
    end
    X = data.(X_field);
    X = X(:);
    
    % Get time points
    times_field = [prefix '_times'];
    time_points = sort(data.(times_field));
    time_points = time_points(:);
    
    nT = length(time_points);
    nX = length(X);
    total_N = nT * nX;
    
    input_list = zeros(total_N, 4, 'single');
    cs_list    = zeros(total_N, 1, 'single');
    ce_list    = zeros(total_N, 1, 'single');
    phie_list  = zeros(total_N, 1, 'single');
    phis_list  = zeros(total_N, 1, 'single');
    JLi_list   = zeros(total_N, 1, 'single');
    
    for ti = 1:nT
        t_val = time_points(ti);
        t_prefix = sprintf('%s_t%d', prefix, t_val);
        
        cs_data  = single(data.([t_prefix '_cs'])(:));
        ce_data  = single(data.([t_prefix '_ce'])(:));
        phie_data = single(data.([t_prefix '_pe'])(:));
        phis_data = single(data.([t_prefix '_ps'])(:));
        JLi_data = single(data.([t_prefix '_J_Li'])(:));
        
        idx_range = (ti-1)*nX + (1:nX);
        input_list(idx_range, :) = [single(X), single(repmat(t_val, nX, 1)), ...
            single(repmat(c_rate, nX, 1)), single(repmat(temperature, nX, 1))];
        cs_list(idx_range)    = cs_data;
        ce_list(idx_range)    = ce_data;
        phie_list(idx_range)  = phie_data;
        phis_list(idx_range)  = phis_data;
        JLi_list(idx_range)   = JLi_data;
    end
    
    input_tensor = single(input_list);
    cs_tensor    = single(cs_list);
    ce_tensor    = single(ce_list);
    phie_tensor  = single(phie_list);
    phis_tensor  = single(phis_list);
    JLi_tensor   = single(JLi_list);
    
    %% Normalize inputs and targets
    input_norm = input_tensor;
    input_norm(:,1) = zscoreNormalize(input_norm(:,1), ns.x_mean, ns.x_std);
    input_norm(:,2) = zscoreNormalize(input_norm(:,2), ns.t_mean, ns.t_std);
    input_norm(:,3) = zscoreNormalize(input_norm(:,3), ns.C_mean, ns.C_std);
    input_norm(:,4) = zscoreNormalize(input_norm(:,4), ns.T_mean, ns.T_std);
    
    cs_norm   = zscoreNormalize(cs_tensor, ns.cs_mean, ns.cs_std);
    ce_norm   = zscoreNormalize(ce_tensor, ns.ce_mean, ns.ce_std);
    phie_norm = zscoreNormalize(phie_tensor, ns.phie_mean, ns.phie_std);
    phis_norm = zscoreNormalize(phis_tensor, ns.phis_mean, ns.phis_std);
    JLi_norm  = zscoreNormalize(JLi_tensor, ns.JLi_mean, ns.JLi_std);
    
    %% Forward pass (no gradients needed)
    inputs_dl = dlarray(input_norm);
    outputs_dl = forwardPINN(params, inputs_dl);
    outputs_norm = extractdata(outputs_dl);
    
    cs_pred_norm   = outputs_norm(:, 1);
    ce_pred_norm   = outputs_norm(:, 2);
    phie_pred_norm = outputs_norm(:, 3);
    phis_pred_norm = outputs_norm(:, 4);
    JLi_pred_norm  = outputs_norm(:, 5);
    
    %% Compute overall metrics (normalized space)
    all_true_norm = [cs_norm, ce_norm, phie_norm, phis_norm, JLi_norm];
    all_pred_norm = [cs_pred_norm, ce_pred_norm, phie_pred_norm, phis_pred_norm, JLi_pred_norm];
    
    rmse = sqrt(mean((all_true_norm(:) - all_pred_norm(:)).^2));
    mae  = mean(abs(all_true_norm(:) - all_pred_norm(:)));
    ss_tot = sum((all_true_norm(:) - mean(all_true_norm(:))).^2);
    ss_res = sum((all_true_norm(:) - all_pred_norm(:)).^2);
    r2 = 1 - ss_res / (ss_tot + 1e-8);
    
    fprintf('Metrics (Normalized Space) for T=%.0f, C=%g:\n', temperature, c_rate);
    fprintf('RMSE: %.4f, MAE: %.4f, R2: %.4f\n', rmse, mae, r2);
    
    %% Denormalize phis for discharge curve
    phis_pred = zscoreDenormalize(phis_pred_norm, ns.phis_mean, ns.phis_std);
    phis_true = phis_tensor;
    
    x_values    = input_tensor(:, 1);
    time_values = input_tensor(:, 2);
    unique_times = unique(time_values);
    unique_times = sort(unique_times);
    
    voltage_pred = zeros(length(unique_times), 1);
    voltage_true = zeros(length(unique_times), 1);
    voltage_mae  = zeros(length(unique_times), 1);
    
    for ti = 1:length(unique_times)
        t_val = unique_times(ti);
        mask = time_values == t_val;
        x_at_time = x_values(mask);
        phis_pred_t = phis_pred(mask);
        phis_true_t = phis_true(mask);
        
        [~, x_end_idx] = max(x_at_time);
        
        voltage_pred(ti) = phis_pred_t(x_end_idx) - bp.R_c * bp.I;
        voltage_true(ti) = phis_true_t(x_end_idx) - bp.R_c * bp.I;
        voltage_mae(ti)  = abs(voltage_pred(ti) - voltage_true(ti));
    end
    
    %% Plot Discharge Curve
    figure('Position', [100, 100, 1000, 600], 'Color', 'w');
    
    plot(unique_times, voltage_true, 'Color', [0.2, 0.2, 0.2], ...
        'LineWidth', 1.5, 'DisplayName', 'True Voltage');
    hold on;
    plot(unique_times, voltage_pred, 'Color', [0.94, 0.5, 0.5], ...
        'LineWidth', 1.5, 'DisplayName', 'Predicted Voltage');
    
    % Add markers at regular intervals
    markerIdx = 1:50:length(unique_times);
    plot(unique_times(markerIdx), voltage_pred(markerIdx), '*', ...
        'Color', [0.94, 0.5, 0.5], 'MarkerSize', 8, 'HandleVisibility', 'off');
    
    xlabel('Time (s)', 'FontSize', 12, 'Color', [0.27, 0.51, 0.71]);
    ylabel('Voltage (V)', 'FontSize', 12, 'Color', [0.27, 0.51, 0.71]);
    title(sprintf('Discharge Curve (T=%d°C, C=%g)', temperature - 273, c_rate), ...
        'FontSize', 14, 'Color', [0.27, 0.51, 0.71]);
    legend('Location', 'best', 'FontSize', 10);
    grid on;
    set(gca, 'GridColor', [0.68, 0.85, 0.9], 'GridLineStyle', '--', 'GridAlpha', 0.5);
    
    saveas(gcf, sprintf('discharge_T%d_C%g.png', temperature - 273, c_rate));
    
    %% Plot Discharge MAE
    figure('Position', [100, 100, 1000, 600], 'Color', 'w');
    
    plot(unique_times, voltage_mae, 'Color', [0.94, 0.5, 0.5], ...
        'LineWidth', 1.2, 'Marker', 'o', 'MarkerSize', 3, ...
        'DisplayName', 'Mean Absolute Error');
    
    xlabel('Time (s)', 'FontSize', 12, 'Color', [0.27, 0.51, 0.71]);
    ylabel('Mean Absolute Error (V)', 'FontSize', 12, 'Color', [0.27, 0.51, 0.71]);
    title(sprintf('MAE of Predicted Voltage (T=%d°C, C=%g)', temperature - 273, c_rate), ...
        'FontSize', 14, 'Color', [0.27, 0.51, 0.71]);
    legend('Location', 'best', 'FontSize', 10);
    grid on;
    set(gca, 'GridColor', [0.68, 0.85, 0.9], 'GridLineStyle', '--', 'GridAlpha', 0.5);
    
    saveas(gcf, sprintf('discharge_mae_T%d_C%g.png', temperature - 273, c_rate));
    
    %% SOC Curve
    cs_pred = zscoreDenormalize(cs_pred_norm, ns.cs_mean, ns.cs_std);
    cs_true = cs_tensor;
    
    x_phys = zscoreDenormalize(input_norm(:,1), ns.x_mean, ns.x_std);
    
    soc_pred = zeros(length(unique_times), 1);
    soc_true = zeros(length(unique_times), 1);
    soc_mae  = zeros(length(unique_times), 1);
    
    for ti = 1:length(unique_times)
        t_val = unique_times(ti);
        mask = time_values == t_val;
        x_at_time = x_phys(mask);
        
        cs_pred_t = cs_pred(mask);
        cs_true_t = cs_true(mask);
        
        % Select anode region
        anode_mask = x_at_time <= bp.L_neg;
        cs_pred_anode = cs_pred_t(anode_mask);
        cs_true_anode = cs_true_t(anode_mask);
        
        % Mean concentration in anode
        mean_cs_pred = mean(cs_pred_anode);
        mean_cs_true = mean(cs_true_anode);
        
        % Stoichiometry
        theta_pred = mean_cs_pred / bp.cs_neg_max;
        theta_true = mean_cs_true / bp.cs_neg_max;
        
        % SOC (using reference stoichiometries from paper)
        theta_0pct   = 0.007;
        theta_100pct = 0.792;
        soc_pred(ti) = (theta_pred - theta_0pct) * 100 / (theta_100pct - theta_0pct);
        soc_true(ti) = (theta_true - theta_0pct) * 100 / (theta_100pct - theta_0pct);
        soc_mae(ti)  = abs(soc_pred(ti) - soc_true(ti));
    end
    
    %% Plot SOC
    figure('Position', [100, 100, 1000, 600], 'Color', 'w');
    
    plot(unique_times, soc_true, 'Color', [0.2, 0.2, 0.2], ...
        'LineWidth', 1.5, 'DisplayName', 'True SOC');
    hold on;
    plot(unique_times, soc_pred, 'Color', [0.94, 0.5, 0.5], ...
        'LineWidth', 1.5, 'DisplayName', 'Predicted SOC');
    
    markerIdx = 1:50:length(unique_times);
    plot(unique_times(markerIdx), soc_pred(markerIdx), '*', ...
        'Color', [0.94, 0.5, 0.5], 'MarkerSize', 8, 'HandleVisibility', 'off');
    
    xlabel('Time (s)', 'FontSize', 12, 'Color', [0.18, 0.55, 0.34]);
    ylabel('SOC (%)', 'FontSize', 12, 'Color', [0.18, 0.55, 0.34]);
    title(sprintf('State of Charge (T=%d°C, C=%g)', temperature - 273, c_rate), ...
        'FontSize', 14, 'Color', [0.18, 0.55, 0.34]);
    legend('Location', 'best', 'FontSize', 10);
    grid on;
    set(gca, 'GridColor', [0.56, 0.93, 0.56], 'GridLineStyle', '--', 'GridAlpha', 0.5);
    
    saveas(gcf, sprintf('soc_T%d_C%g.png', temperature - 273, c_rate));
    
    %% Plot SOC MAE
    figure('Position', [100, 100, 1000, 600], 'Color', 'w');
    
    plot(unique_times, soc_mae, 'Color', [0.94, 0.5, 0.5], ...
        'LineWidth', 1.2, 'Marker', 'o', 'MarkerSize', 3, ...
        'DisplayName', 'Mean Absolute Error');
    
    xlabel('Time (s)', 'FontSize', 12, 'Color', [0.18, 0.55, 0.34]);
    ylabel('Mean Absolute Error (%)', 'FontSize', 12, 'Color', [0.18, 0.55, 0.34]);
    title(sprintf('MAE of SOC (T=%d°C, C=%g)', temperature - 273, c_rate), ...
        'FontSize', 14, 'Color', [0.18, 0.55, 0.34]);
    legend('Location', 'best', 'FontSize', 10);
    grid on;
    set(gca, 'GridColor', [0.56, 0.93, 0.56], 'GridLineStyle', '--', 'GridAlpha', 0.5);
    
    saveas(gcf, sprintf('soc_mae_T%d_C%g.png', temperature - 273, c_rate));
    
    fprintf('Evaluation plots saved.\n');
end
